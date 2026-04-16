from __future__ import annotations

from types import SimpleNamespace

from app.services.governance.launch_gates import GateVerdict
from scripts import run_launch_gates as cli


def test_cli_passes_provenance_fields_to_runner(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def fake_run_all_gates(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            verdict=GateVerdict.PASS,
            hard_gate_count=9,
            hard_passed=9,
            hard_failed=0,
            hard_missing_evidence=0,
            enabled_markets=kwargs.get("enabled_markets") or ["US", "HK", "JP", "TW"],
            target_market=kwargs.get("target_market"),
            execution_mode=kwargs.get("execution_mode"),
            provenance_note=kwargs.get("provenance_note"),
            content_hash="abc123",
        )

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
            "run_launch_gates.py",
            "--no-db",
            "--enabled-market",
            "US",
            "--enabled-market",
            "HK",
            "--target-market",
            "HK",
            "--execution-mode",
            "synthetic_seeded_harness",
            "--provenance-note",
            "Synthetic rehearsal only.",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert captured["enabled_markets"] == ["US", "HK"]
    assert captured["target_market"] == "HK"
    assert captured["execution_mode"] == "synthetic_seeded_harness"
    assert captured["provenance_note"] == "Synthetic rehearsal only."

    out = capsys.readouterr().out
    assert "Enabled markets: ['US', 'HK']" in out
    assert "Target market: HK" in out
    assert "Execution mode: synthetic_seeded_harness" in out
    assert "Provenance: Synthetic rehearsal only." in out

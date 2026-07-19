from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_export_static_market_artifact_writes_status_and_preserves_exit_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.scripts import export_static_market_artifact

    forwarded_args = []

    def fake_export_main(argv):
        forwarded_args.extend(argv)
        return 79

    monkeypatch.setattr(export_static_market_artifact.export_static_site, "main", fake_export_main)

    result = export_static_market_artifact.main(
        [
            "--output-dir",
            str(tmp_path),
            "--refresh-daily",
            "--market",
            "CN",
        ]
    )

    assert result == 79
    assert forwarded_args == [
        "--output-dir",
        str(tmp_path),
        "--refresh-daily",
        "--market",
        "CN",
    ]
    assert json.loads((tmp_path / "status" / "cn" / "status.json").read_text()) == {
        "market": "CN",
        "has_current_artifact": False,
        "status": "failed",
        "reason": "no_current_artifact",
    }


def test_export_static_market_artifact_writes_success_status(monkeypatch, tmp_path: Path) -> None:
    from app.scripts import export_static_market_artifact

    monkeypatch.setattr(export_static_market_artifact.export_static_site, "main", lambda argv: 0)

    result = export_static_market_artifact.main(
        [
            "--output-dir",
            str(tmp_path),
            "--market",
            "CN",
        ]
    )

    assert result == 0
    assert json.loads((tmp_path / "status" / "cn" / "status.json").read_text()) == {
        "market": "CN",
        "has_current_artifact": True,
        "status": "published",
        "reason": None,
    }


def test_export_static_market_artifact_writes_skipped_status_for_not_trading_day(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.scripts import export_static_market_artifact

    monkeypatch.setattr(
        export_static_market_artifact.export_static_site,
        "main",
        lambda argv: export_static_market_artifact.export_static_site.STATIC_EXPORT_SKIPPED_EXIT_CODE,
    )

    result = export_static_market_artifact.main(
        [
            "--output-dir",
            str(tmp_path),
            "--market",
            "CN",
        ]
    )

    assert result == export_static_market_artifact.export_static_site.STATIC_EXPORT_SKIPPED_EXIT_CODE
    assert json.loads((tmp_path / "status" / "cn" / "status.json").read_text()) == {
        "market": "CN",
        "has_current_artifact": False,
        "status": "skipped",
        "reason": "not_trading_day",
    }


def test_export_static_market_artifact_writes_failed_status_when_export_raises(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.scripts import export_static_market_artifact

    def fail_export(_argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(export_static_market_artifact.export_static_site, "main", fail_export)

    with pytest.raises(RuntimeError, match="boom"):
        export_static_market_artifact.main(
            [
                "--output-dir",
                str(tmp_path),
                "--market",
                "CN",
            ]
        )

    assert json.loads((tmp_path / "status" / "cn" / "status.json").read_text()) == {
        "market": "CN",
        "has_current_artifact": False,
        "status": "failed",
        "reason": "export_failed",
    }


def test_export_static_market_artifact_writes_failed_status_when_export_exits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app.scripts import export_static_market_artifact

    def exit_export(_argv):
        raise SystemExit("--combine-artifacts-dir cannot be used together with --refresh-daily")

    monkeypatch.setattr(export_static_market_artifact.export_static_site, "main", exit_export)

    with pytest.raises(SystemExit):
        export_static_market_artifact.main(
            [
                "--output-dir",
                str(tmp_path),
                "--market",
                "CN",
            ]
        )

    assert json.loads((tmp_path / "status" / "cn" / "status.json").read_text()) == {
        "market": "CN",
        "has_current_artifact": False,
        "status": "failed",
        "reason": "export_failed",
    }

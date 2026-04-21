from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import app.scripts.build_daily_price_bundle as build_script
import app.scripts.sync_daily_price_bundle_from_github as sync_daily_script
import app.scripts.sync_weekly_reference_from_github as sync_weekly_script


@contextmanager
def _fake_session(db="db-session"):
    yield db


def test_build_daily_price_bundle_exports_market_bundle(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(build_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(build_script, "SessionLocal", _fake_session)

    def _fake_export(db, **kwargs):
        _ = db
        kwargs["latest_manifest_path"].write_text(
            json.dumps({"bundle_asset_name": kwargs["bundle_asset_name"]}),
            encoding="utf-8",
        )
        kwargs["output_path"].write_bytes(b"bundle")
        return {
            "bundle_path": str(kwargs["output_path"]),
            "manifest_path": str(kwargs["latest_manifest_path"]),
            "market": kwargs["market"],
            "as_of_date": kwargs["as_of_date"].isoformat(),
            "symbol_count": 2,
            "bar_period": "2y",
        }

    service = SimpleNamespace(
        market_calendar=SimpleNamespace(
            last_completed_trading_day=lambda market: date(2026, 4, 21)
        ),
        latest_manifest_name_for_market=lambda market: f"daily-price-latest-{market.lower()}.json",
        export_daily_price_bundle=_fake_export,
    )
    monkeypatch.setattr(build_script, "get_daily_price_bundle_service", lambda: service)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_daily_price_bundle",
            "--market",
            "US",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert build_script.main() == 0
    stdout = capsys.readouterr().out
    assert "Daily price bundle complete for US:" in stdout
    assert (tmp_path / "daily-price-us-20260421.json.gz").exists()
    assert (tmp_path / "daily-price-latest-us.json").exists()


def test_sync_weekly_reference_from_github_script_calls_service(monkeypatch, capsys):
    monkeypatch.setattr(sync_weekly_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_weekly_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_weekly_script,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            sync_weekly_reference_from_github=lambda db, **kwargs: {
                "status": "success",
                "market": kwargs["market"],
                "source_revision": "fundamentals_v1_us:20260418120000",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_weekly_reference_from_github", "--market", "US"],
    )

    assert sync_weekly_script.main() == 0
    assert "Weekly GitHub sync result:" in capsys.readouterr().out


def test_sync_daily_price_bundle_from_github_script_calls_service(monkeypatch, capsys):
    monkeypatch.setattr(sync_daily_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(sync_daily_script, "SessionLocal", _fake_session)
    monkeypatch.setattr(
        sync_daily_script,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(
            sync_from_github=lambda db, **kwargs: {
                "status": "success",
                "market": kwargs["market"],
                "source_revision": "daily_prices_us:20260421120000",
            }
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sync_daily_price_bundle_from_github", "--market", "US"],
    )

    assert sync_daily_script.main() == 0
    assert "Daily GitHub price sync result:" in capsys.readouterr().out

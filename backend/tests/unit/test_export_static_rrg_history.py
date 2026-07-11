"""Focused CLI coverage for rolling static RRG state."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest

from app.domain.markets import market_registry
from app.domain.markets.catalog import get_market_catalog
from app.scripts import describe_static_rrg_history
from app.scripts import export_static_site as export_script
from app.services.static_rrg_history_contract import (
    build_static_rrg_history_plan,
    static_rrg_asset_name,
)


class _SessionFactory:
    def __call__(self):
        return self

    def __enter__(self):
        return object()

    def __exit__(self, *_args):
        return False


def _export_result(output_dir):
    return SimpleNamespace(
        output_dir=output_dir,
        generated_at="2026-07-10T22:00:00Z",
        as_of_date="2026-07-10",
        warnings=(),
        manifest={},
    )


@pytest.mark.parametrize("market", market_registry.supported_market_codes())
def test_rrg_automation_plan_matches_live_market_catalog(market, tmp_path):
    catalog = get_market_catalog()
    plan = build_static_rrg_history_plan(
        market=market,
        directory=tmp_path,
        market_catalog=catalog,
    )

    assert plan.enabled is bool(catalog.rrg_scopes_for_market(market))
    assert plan.asset_name == static_rrg_asset_name(market)
    assert plan.source_path == tmp_path / plan.asset_name
    assert plan.output_path == tmp_path / "current" / plan.asset_name


def test_describe_rrg_history_emits_machine_readable_plan(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "describe_static_rrg_history.py",
            "--market",
            "HK",
            "--directory",
            str(tmp_path),
        ],
    )

    assert describe_static_rrg_history.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == build_static_rrg_history_plan(
        market="HK",
        directory=tmp_path,
    ).as_dict()


def test_describe_rrg_history_does_not_require_database_runtime(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.scripts.describe_static_rrg_history",
            "--market",
            "HK",
            "--directory",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "DATABASE_URL": "sqlite:///tmp/rrg-describe.db"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["enabled"] is True


def test_main_loads_advances_and_persists_market_rrg_state(monkeypatch, tmp_path):
    calls = []
    state = SimpleNamespace(through_date=date(2026, 7, 10))
    history_path = tmp_path / "history" / "rrg-history-hk.json.gz"
    history_path.parent.mkdir()
    history_path.write_bytes(b"existing")

    class _HistoryService:
        def enabled_for_market(self, market):
            return market == "HK"

        def prepare(self, db, *, market, through_date, directory):
            calls.append(("prepare", db, market, through_date, directory))
            return SimpleNamespace(state=state, warnings=())

        def persist(self, preparation, *, exported_as_of_date):
            calls.append(("persist", preparation.state, exported_as_of_date))
            return {"weeks": 12}

    class _ExportService:
        def __init__(self, _session_factory, *, rrg_payload_source):
            calls.append(("service", rrg_payload_source.history_state))

        def export(self, output_dir, **_kwargs):
            calls.append(("export", output_dir))
            return _export_result(output_dir)

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(export_script, "SessionLocal", _SessionFactory())
    monkeypatch.setattr(export_script, "StaticRRGHistoryBundleService", _HistoryService)
    monkeypatch.setattr(export_script, "StaticSiteExportService", _ExportService)
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--market",
            "HK",
            "--rrg-history-dir",
            str(history_path.parent),
        ],
    )

    assert export_script.main() == 0
    assert [call[0] for call in calls] == ["prepare", "service", "export", "persist"]
    assert calls[0][2:] == ("HK", date(2026, 7, 10), history_path.parent)
    assert calls[1][1] is state


def test_main_does_not_fail_when_rrg_state_cannot_be_persisted(
    monkeypatch,
    tmp_path,
    capsys,
):
    state = SimpleNamespace(through_date=date(2026, 7, 10))

    class _HistoryService:
        def enabled_for_market(self, _market):
            return True

        def prepare(self, *_args, **_kwargs):
            return SimpleNamespace(state=state, warnings=())

        def persist(self, *_args, **_kwargs):
            raise export_script.StaticRRGHistoryBundleError("disk full")

    class _ExportService:
        def __init__(self, _session_factory, **_kwargs):
            pass

        def export(self, output_dir, **_kwargs):
            return _export_result(output_dir)

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(export_script, "SessionLocal", _SessionFactory())
    monkeypatch.setattr(export_script, "StaticRRGHistoryBundleService", _HistoryService)
    monkeypatch.setattr(export_script, "StaticSiteExportService", _ExportService)
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda _market: date(2026, 7, 10),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--market",
            "HK",
            "--rrg-history-dir",
            str(tmp_path / "history"),
        ],
    )

    assert export_script.main() == 0
    assert "Rolling RRG history was not persisted: disk full" in capsys.readouterr().out

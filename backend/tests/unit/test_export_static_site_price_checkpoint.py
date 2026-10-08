"""The static export stops at its price-stage deadline and checkpoints (#502)."""

from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import date

import pytest

import app.scripts.export_static_site as export_script


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--output-dir",
        str(tmp_path / "out"),
        "--refresh-daily",
        "--market",
        "US",
        *extra,
    ]


def test_main_checkpoints_a_resumable_price_stage_and_skips_derived_work(
    monkeypatch, tmp_path, capsys
):
    refresh_kwargs: dict = {}
    checkpoints: list[dict] = []

    def run_daily_refresh(**kwargs):
        refresh_kwargs.update(kwargs)
        return (
            {
                "price_refresh": {
                    "status": "resumable",
                    "market": "US",
                    "as_of_date": "2026-04-17",
                }
            },
            [],
        )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("no derived export after a resumable price stage")

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(export_script, "_run_daily_refresh", run_daily_refresh)
    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(export_script, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        export_script,
        "write_price_checkpoint",
        lambda db, **kwargs: checkpoints.append(kwargs)
        or {"bundle_asset_name": "price-checkpoint-us.json.gz", "symbol_count": 2},
    )
    started = time.monotonic()

    exit_code = export_script.main(
        _argv(
            tmp_path,
            "--price-stage-deadline-minutes",
            "2",
            "--price-checkpoint-dir",
            str(tmp_path / "checkpoint"),
        )
    )

    assert exit_code == export_script.STATIC_EXPORT_PRICE_CHECKPOINTED_EXIT_CODE == 80
    assert checkpoints == [
        {
            "market": "US",
            "as_of_date": date(2026, 4, 17),
            "output_dir": tmp_path / "checkpoint",
        }
    ]
    deadline = refresh_kwargs["price_stage_deadline"]
    assert started + 115 < deadline <= time.monotonic() + 120
    assert "Re-run" in capsys.readouterr().out


def test_main_runs_without_a_deadline_by_default(monkeypatch, tmp_path):
    refresh_kwargs: dict = {}

    def run_daily_refresh(**kwargs):
        refresh_kwargs.update(kwargs)
        raise RuntimeError("stop after capturing the refresh arguments")

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(export_script, "_run_daily_refresh", run_daily_refresh)

    with pytest.raises(RuntimeError, match="stop after"):
        export_script.main(_argv(tmp_path))

    assert refresh_kwargs["price_stage_deadline"] is None


@pytest.mark.parametrize(
    "extra",
    [
        ("--price-stage-deadline-minutes", "2"),
        ("--price-checkpoint-dir", "/tmp/checkpoint"),
    ],
)
def test_main_requires_deadline_and_checkpoint_dir_together(tmp_path, extra):
    with pytest.raises(SystemExit):
        export_script.main(_argv(tmp_path, *extra))


def test_daily_refresh_returns_before_market_rs_when_prices_are_resumable(monkeypatch):
    price_calls: list[dict] = []
    monkeypatch.setattr(export_script, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(export_script, "disable_serialized_data_fetch_lock", nullcontext)
    monkeypatch.setattr(export_script, "disable_serialized_market_workload", nullcontext)
    monkeypatch.setattr(export_script, "_tracked_ibd_csv_path", lambda: "ibd.csv")
    monkeypatch.setattr(
        export_script.IBDIndustryService, "load_from_csv", lambda _db, csv_path: 0
    )
    monkeypatch.setattr(
        export_script,
        "_resolve_latest_completed_trading_date",
        lambda market: date(2026, 4, 17),
    )
    monkeypatch.setattr(
        export_script,
        "_refresh_static_daily_prices",
        lambda **kwargs: price_calls.append(kwargs)
        or {"status": "resumable", "market": "US", "as_of_date": "2026-04-17"},
    )
    monkeypatch.setattr(
        export_script,
        "_prepare_static_rs_formula",
        lambda **_kwargs: pytest.fail("Market RS must wait for a complete price stage"),
    )

    results, _warnings = export_script._run_daily_refresh(
        market="US",
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
        skip_cot_refresh=True,
        price_stage_deadline=123.0,
    )

    assert price_calls == [
        {
            "as_of_date": date(2026, 4, 17),
            "market": "US",
            "repair_price_history": False,
            "deadline": 123.0,
        }
    ]
    assert results["price_refresh"]["status"] == "resumable"


def test_refresh_static_daily_prices_passes_the_deadline_to_the_service(monkeypatch):
    init_kwargs: dict = {}

    class FakeService:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)

        def refresh(self, **_kwargs):
            return {"status": "completed"}

    monkeypatch.setattr(export_script, "StaticDailyPriceRefreshService", FakeService)
    monkeypatch.setattr(export_script, "SessionLocal", object())
    monkeypatch.setattr(export_script, "get_price_cache", lambda: object())
    monkeypatch.setattr(export_script, "BulkDataFetcher", lambda: object())

    export_script._refresh_static_daily_prices(
        as_of_date=date(2026, 4, 17), market="US", deadline=99.0
    )

    assert init_kwargs["deadline"] == 99.0

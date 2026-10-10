"""Tests for static-site no-current artifact CLI outcomes."""

from __future__ import annotations

import json
import shutil
import sys

import pytest

import app.scripts.export_static_site as export_script


def test_main_returns_skip_code_for_market_not_trading_day(monkeypatch, tmp_path, capsys):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "TW": {
                        "status": "skipped",
                        "reason": "not_trading_day",
                        "market": "TW",
                        "as_of_date": "2026-05-01",
                    }
                }
            },
            ["Static export market TW snapshot returned status 'skipped' (not_trading_day)."],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("market export should not run for not_trading_day")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "TW",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_SKIPPED_EXIT_CODE

    captured = capsys.readouterr()
    assert "Static site export skipped for market TW because it is not a trading day." in captured.out
    assert export_calls == []
    assert not (output_dir / "diagnostics" / "tw" / "snapshot-failure.json").exists()


def test_main_returns_no_current_artifact_code_for_selected_market_exposure_error(
    monkeypatch,
    tmp_path,
    capsys,
):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "market_exposure": {
                    "IN": {
                        "market": "IN",
                        "date": "2026-06-25",
                        "error": "no_benchmark_data",
                    }
                },
                "feature_snapshots": {
                    "IN": {
                        "status": "skipped",
                        "reason": "market_exposure_not_ready",
                        "market": "IN",
                        "as_of_date": "2026-06-25",
                        "warnings": [
                            "Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."
                        ],
                        "failure_diagnostics": {
                            "date": "2026-06-25",
                            "error": "no_benchmark_data",
                        },
                    }
                },
            },
            ["Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("market export should not run when exposure is missing")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "IN",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    diagnostics_path = output_dir / "diagnostics" / "in" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload == {
        "market": "IN",
        "status": "skipped",
        "reason": "market_exposure_not_ready",
        "failed_symbols": [],
        "row_count": None,
        "warnings": ["Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."],
        "failure_diagnostics": {
            "date": "2026-06-25",
            "error": "no_benchmark_data",
        },
    }
    captured = capsys.readouterr()
    assert "exposure was not stored" in captured.out
    assert "fallback" in captured.out
    assert export_calls == []


def test_main_returns_no_current_artifact_code_for_all_market_exposure_error(
    monkeypatch,
    tmp_path,
    capsys,
):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "market_exposure": {
                    "US": {
                        "market": "US",
                        "date": "2026-06-25",
                        "exposure_score": 75.0,
                    },
                    "IN": {
                        "market": "IN",
                        "date": "2026-06-25",
                        "error": "no_benchmark_data",
                    },
                },
                "feature_snapshots": {
                    "US": {
                        "status": "published",
                        "market": "US",
                        "run_id": 90,
                    },
                    "IN": {
                        "status": "skipped",
                        "reason": "market_exposure_not_ready",
                        "market": "IN",
                        "as_of_date": "2026-06-25",
                        "warnings": [
                            "Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."
                        ],
                        "failure_diagnostics": {
                            "date": "2026-06-25",
                            "error": "no_benchmark_data",
                        },
                    },
                },
            },
            ["Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("all-market export should not run when any exposure is missing")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    diagnostics_path = output_dir / "diagnostics" / "in" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload == {
        "market": "IN",
        "status": "skipped",
        "reason": "market_exposure_not_ready",
        "failed_symbols": [],
        "row_count": None,
        "warnings": ["Static export market IN exposure not stored for 2026-06-25: no_benchmark_data."],
        "failure_diagnostics": {
            "date": "2026-06-25",
            "error": "no_benchmark_data",
        },
    }
    captured = capsys.readouterr()
    assert "market IN" in captured.out
    assert "exposure was not stored" in captured.out
    assert export_calls == []


def test_write_market_diagnostics_records_quarantined_snapshot(tmp_path):
    path = export_script._write_market_diagnostics(  # noqa: SLF001 - intentional unit test coverage
        tmp_path / "out",
        "in",
        {
            "status": "quarantined",
            "reason": "data_quality_gate",
            "run_id": 91,
            "existing_run_id": 77,
            "failed_symbols": ["RELIANCE.NS", "TCS.NS"],
            "row_count": 4312,
            "warnings": ["price rows missing"],
            "failure_diagnostics": {"failed_symbol_count": 2},
            "ignored": "not exported",
        },
    )

    assert path == tmp_path / "out" / "diagnostics" / "in" / "snapshot-failure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "market": "IN",
        "status": "quarantined",
        "reason": "data_quality_gate",
        "run_id": 91,
        "existing_run_id": 77,
        "failed_symbols": ["RELIANCE.NS", "TCS.NS"],
        "row_count": 4312,
        "warnings": ["price rows missing"],
        "failure_diagnostics": {"failed_symbol_count": 2},
    }


def test_main_returns_no_current_artifact_code_for_group_rank_backfill_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "US": {
                        "status": "quarantined",
                        "reason": "group_rank_backfill_not_ready",
                        "market": "US",
                        "run_id": 77,
                        "as_of_date": "2026-07-24",
                        "warnings": [
                            "Static export market US group-rank history backfill not "
                            "ready for 2026-07-24: errored."
                        ],
                        "failure_diagnostics": {
                            "group_rank_history_backfill": {
                                "status": "errored",
                                "market": "US",
                                "as_of_date": "2026-07-24",
                                "lookback_start_date": "2026-01-18",
                                "error": "US current price coverage is 0.0%",
                            }
                        },
                    }
                }
            },
            [
                "Static export market US group-rank history backfill not ready "
                "for 2026-07-24: errored."
            ],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("market export should use fallback after backfill failure")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "US",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    diagnostics_path = output_dir / "diagnostics" / "us" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["market"] == "US"
    assert payload["status"] == "quarantined"
    assert payload["reason"] == "group_rank_backfill_not_ready"
    assert payload["run_id"] == 77
    assert (
        payload["failure_diagnostics"]["group_rank_history_backfill"]["error"]
        == "US current price coverage is 0.0%"
    )
    captured = capsys.readouterr()
    assert "group-rank history backfill was not ready" in captured.out
    assert "fallback" in captured.out
    assert export_calls == []


def test_main_returns_no_current_artifact_code_for_market_rs_price_coverage_gap(
    monkeypatch,
    tmp_path,
    capsys,
):
    export_calls: list[object] = []
    output_dir = tmp_path / "out"
    warning = (
        "Static export market DE Market RS not ready for 2026-07-27: "
        "current_adjusted_price_coverage_below_threshold."
    )

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "DE": {
                        "status": "skipped",
                        "reason": "market_rs_not_ready",
                        "market": "DE",
                        "as_of_date": "2026-07-27",
                        "warnings": [warning],
                        "failure_diagnostics": {
                            "reason_code": (
                                "current_adjusted_price_coverage_below_threshold"
                            ),
                            "diagnostics": {
                                "current_price_coverage": 0.8434065934065934,
                                "minimum_current_price_coverage": 0.88,
                                "current_prices_available": 1228,
                                "expected_symbol_count": 1456,
                            },
                        },
                    }
                }
            },
            [warning],
        ),
    )

    class ExportShouldNotRun:
        def __init__(self, *_args, **_kwargs):
            export_calls.append("constructed")

        def export(self, *_args, **_kwargs):
            raise AssertionError("market export should use fallback when Market RS is not ready")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportShouldNotRun)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "DE",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    diagnostics_path = output_dir / "diagnostics" / "de" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload == {
        "market": "DE",
        "status": "skipped",
        "reason": "market_rs_not_ready",
        "failed_symbols": [],
        "row_count": None,
        "warnings": [warning],
        "failure_diagnostics": {
            "reason_code": "current_adjusted_price_coverage_below_threshold",
            "diagnostics": {
                "current_price_coverage": 0.8434065934065934,
                "minimum_current_price_coverage": 0.88,
                "current_prices_available": 1228,
                "expected_symbol_count": 1456,
            },
        },
    }
    captured = capsys.readouterr()
    assert "Market RS was not ready" in captured.out
    assert "fallback" in captured.out
    assert export_calls == []


def test_no_current_artifact_exit_message_uses_generic_detail_for_unknown_reason():
    message = export_script._no_current_artifact_exit_message(  # noqa: SLF001
        market="DE",
        failed_markets=("DE",),
        reasons=("new_allowlisted_reason",),
    )

    assert "one or more market artifacts were not current" in message
    assert "fallback" in message


def test_main_returns_no_current_artifact_code_for_quarantined_selected_market(
    monkeypatch,
    tmp_path,
    capsys,
):
    output_dir = tmp_path / "out"

    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        # data_quality_gate is not allowlisted by the new collector, so this
        # intentionally falls through to the existing artifact-missing handler.
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "IN": {
                        "status": "quarantined",
                        "reason": "data_quality_gate",
                        "market": "IN",
                        "run_id": 91,
                        "failed_symbols": ["TCS.NS"],
                        "row_count": 4312,
                        "failure_diagnostics": {"failed_symbol_count": 1},
                    }
                }
            },
            [],
        ),
    )

    class ExportRaisesNoCurrentArtifact:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, output_dir_arg, *_args, **_kwargs):
            shutil.rmtree(output_dir_arg, ignore_errors=True)
            raise export_script.NoPublishedStaticMarketArtifact(
                "No current IN artifact",
                markets=("IN",),
            )

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportRaisesNoCurrentArtifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(output_dir),
            "--refresh-daily",
            "--market",
            "IN",
        ],
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    diagnostics_path = output_dir / "diagnostics" / "in" / "snapshot-failure.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["market"] == "IN"
    assert payload["status"] == "quarantined"
    assert payload["failure_diagnostics"] == {"failed_symbol_count": 1}
    captured = capsys.readouterr()
    assert "fallback" in captured.out
    assert "skipped" in captured.out


def test_main_reraises_unrelated_runtime_errors_for_non_ready_selected_market(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        # Missing reason is not allowlisted by the new collector, so this
        # intentionally falls through to the original runtime-error path.
        lambda **_kwargs: (
            {
                "feature_snapshots": {
                    "IN": {
                        "status": "quarantined",
                        "market": "IN",
                        "failure_diagnostics": {"failed_symbol_count": 1},
                    }
                }
            },
            [],
        ),
    )

    class ExportRaisesUnrelatedRuntimeError:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, *_args, **_kwargs):
            raise RuntimeError("database connection dropped")

    monkeypatch.setattr(export_script, "StaticSiteExportService", ExportRaisesUnrelatedRuntimeError)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_static_site.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--refresh-daily",
            "--market",
            "IN",
        ],
    )

    with pytest.raises(RuntimeError, match="database connection dropped"):
        export_script.main()


_AU_REPAIR = {
    "status": "verified",
    "gap_symbols": 1731,
    "repaired_symbols": 1691,
    "unresolved_symbols": 40,
    "unresolved_count_by_date": {"2026-09-10": 15},
}
_AU_HISTORY_GAPS = {
    "symbol_count": 1704,
    "share": 0.92,
    "count_by_anchor": {"2026-09-07": 1695},
    "current_prices_available": 1845,
}


def _run_au(monkeypatch, tmp_path, feature_snapshot, market_rs):
    output_dir = tmp_path / "out"
    monkeypatch.setattr(export_script, "prepare_runtime", lambda: None)
    monkeypatch.setattr(
        export_script,
        "_run_daily_refresh",
        lambda **_kwargs: (
            {
                "price_refresh": {
                    "status": "completed",
                    "market": "AU",
                    "as_of_date": "2026-10-08",
                    "yahoo_fetched_symbols": 1731,
                    "yahoo_failed_symbols": 40,
                    "rs_anchor_repair": _AU_REPAIR,
                },
                "market_rs": {"AU": market_rs},
                "feature_snapshots": {"AU": feature_snapshot},
            },
            [],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_static_site.py", "--output-dir", str(output_dir), "--refresh-daily", "--market", "AU"],
    )
    return output_dir


def test_main_records_rs_coverage_when_anchor_gaps_reject_rs(monkeypatch, tmp_path):
    """#539: a market kept on its last good artifact still reports why."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output_dir = _run_au(
        monkeypatch,
        tmp_path,
        {
            "status": "skipped",
            "reason": "market_rs_not_ready",
            "market": "AU",
            "as_of_date": "2026-10-08",
            "failure_diagnostics": {"reason_code": "historical_adjusted_anchor_gap_above_threshold"},
        },
        {
            "status": "failed",
            "reason_code": "historical_adjusted_anchor_gap_above_threshold",
            "eligible_symbol_count": 84,
            "history_gaps": _AU_HISTORY_GAPS,
        },
    )

    assert export_script.main() == export_script.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE

    coverage = json.loads((output_dir / "diagnostics" / "au" / "coverage.json").read_text())
    assert coverage == {
        "market": "AU",
        "as_of_date": "2026-10-08",
        "seed_source_revision": None,
        "expected_universe": None,
        "latest_session_priced": 1845,
        "rs_status": "failed",
        "rs_reason": "historical_adjusted_anchor_gap_above_threshold",
        "rs_eligible": 84,
        "scored": None,
        "snapshot_status": "skipped",
        "snapshot_reason": "market_rs_not_ready",
        "fetched_symbols": 1731,
        "failed_symbols": 40,
        "rs_anchor_repair": _AU_REPAIR,
        "history_gaps": _AU_HISTORY_GAPS,
    }
    assert (
        "| AU | 2026-10-08 | 1845 | 84 | — | failed: historical_adjusted_anchor_gap_above_threshold "
        "| 1691/1731 repaired, 40 unresolved |"
    ) in summary.read_text()


def test_main_records_scored_count_after_a_cleaning_export(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        export_script,
        "_snapshot_score_counts",
        lambda market, run_id: (
            (1766, "daily_prices_au:20261008") if (market, run_id) == ("AU", 7) else (None, None)
        ),
    )
    output_dir = _run_au(
        monkeypatch,
        tmp_path,
        {"status": "published", "market": "AU", "run_id": 7, "total_symbols": 2101},
        {"status": "completed", "eligible_symbol_count": 1766, "history_gaps": _AU_HISTORY_GAPS},
    )

    class CleaningExport:
        def __init__(self, *_args, **_kwargs):
            pass

        def export(self, output_dir_arg, **_kwargs):
            # The real export wipes its output directory first.
            shutil.rmtree(output_dir_arg, ignore_errors=True)
            output_dir_arg.mkdir(parents=True)
            return type(
                "Result",
                (),
                {"output_dir": output_dir_arg, "generated_at": "now", "as_of_date": "2026-10-08", "warnings": ()},
            )()

    monkeypatch.setattr(export_script, "StaticSiteExportService", CleaningExport)
    assert export_script.main() == 0

    coverage = json.loads((output_dir / "diagnostics" / "au" / "coverage.json").read_text())
    assert coverage["scored"] == 1766
    assert coverage["expected_universe"] == 2101
    assert coverage["seed_source_revision"] == "daily_prices_au:20261008"
    assert coverage["rs_eligible"] == 1766


def test_snapshot_score_counts_reads_scored_rows_and_seed_revision(db_session):
    from datetime import date

    from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
    from app.models.app_settings import AppSetting
    from app.services.daily_price_bundle_service import DailyPriceBundleService

    day = date(2026, 10, 8)
    run = FeatureRun(as_of_date=day, run_type="daily_snapshot", status="published")
    other = FeatureRun(as_of_date=day, run_type="daily_snapshot", status="published")
    db_session.add_all([run, other])
    db_session.flush()
    db_session.add_all(
        [
            StockFeatureDaily(run_id=run.id, symbol="BHP.AX", as_of_date=day, composite_score=71.0),
            StockFeatureDaily(run_id=run.id, symbol="CBA.AX", as_of_date=day, composite_score=None),
            StockFeatureDaily(run_id=other.id, symbol="BHP.AX", as_of_date=day, composite_score=50.0),
            AppSetting(
                key=DailyPriceBundleService.sync_state_key("AU"),
                value=json.dumps({"source_revision": "daily_prices_au:20261008"}),
            ),
        ]
    )
    db_session.commit()

    assert export_script._snapshot_score_counts("AU", run.id) == (  # noqa: SLF001
        1,
        "daily_prices_au:20261008",
    )

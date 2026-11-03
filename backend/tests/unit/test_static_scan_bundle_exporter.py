"""Characterization tests for static scan bundle export."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from app.services.static_scan_bundle_exporter import (
    StaticScanBundleExporter,
    StaticScanBundleRequest,
)


def test_bundle_preserves_compact_projection_and_manifest(tmp_path: Path):
    run = SimpleNamespace(
        id=42,
        as_of_date=date(2026, 8, 21),
        config_json={"materialization_versions": {"opportunity_state": 1}},
    )
    rows = [SimpleNamespace(symbol="NVDA")]
    publication = SimpleNamespace(
        snapshot=SimpleNamespace(
            formula_version="balanced-v1",
            as_of_date=date(2026, 8, 21),
        ),
        market_rs_run_id=9,
        universe_size=5000,
    )
    exporter = StaticScanBundleExporter(
        rs_identity_resolver=lambda *_args, **_kwargs: SimpleNamespace(
            publication=publication
        )
    )

    result = exporter.export_scan_bundle(
        StaticScanBundleRequest(
            output_dir=tmp_path,
            generated_at="2026-08-22T00:00:00Z",
            run=run,
            rows=rows,
            filter_options=SimpleNamespace(
                ibd_industries=(),
                gics_sectors=("Technology",),
                ratings=("Buy",),
            ),
            market="US",
            row_serializer=lambda row: {
                "symbol": row.symbol,
                "composite_score": 88.0,
                "volume": 200_000_000,
                "action_state": "setup_ready",
                "opportunity_state": {"score_pillars": {"trend_integrity": 20.0}},
            },
        )
    )

    chunk = json.loads((tmp_path / "scan/chunks/chunk-0001.json").read_text())
    row = chunk["rows"][0]
    assert row["action_state"] == "setup_ready"
    assert row["opportunity_state"]["score_pillars"]["trend_integrity"] == 20.0
    assert "setup_engine" not in row
    assert result.manifest["features"] == {"opportunity_state": True}
    assert result.manifest["rows_total"] == 1
    assert result.serialized_rows == (row,)


def test_bundle_ranks_fully_scored_rows_before_ipo_only_rows(tmp_path: Path):
    scores = {
        "IPO": (72.0, None, None),
        "BEST": (65.0, 65.0, 62.0),
        "CANSLIM": (60.0, None, 60.0),
    }
    publication = SimpleNamespace(
        snapshot=SimpleNamespace(formula_version="balanced-v1", as_of_date=date(2026, 8, 21)),
        market_rs_run_id=9,
        universe_size=5000,
    )
    exporter = StaticScanBundleExporter(
        rs_identity_resolver=lambda *_args, **_kwargs: SimpleNamespace(
            publication=publication
        )
    )

    result = exporter.export_scan_bundle(
        StaticScanBundleRequest(
            output_dir=tmp_path,
            generated_at="2026-08-22T00:00:00Z",
            run=SimpleNamespace(id=42, as_of_date=date(2026, 8, 21), config_json={}),
            rows=[SimpleNamespace(symbol=symbol) for symbol in scores],
            filter_options=SimpleNamespace(ibd_industries=(), gics_sectors=(), ratings=()),
            market="US",
            row_serializer=lambda row: {
                "symbol": row.symbol,
                "composite_score": scores[row.symbol][0],
                "minervini_score": scores[row.symbol][1],
                "canslim_score": scores[row.symbol][2],
                "volume": 200_000_000,
            },
        )
    )

    assert [row["symbol"] for row in result.serialized_rows] == ["BEST", "CANSLIM", "IPO"]
    assert [row["symbol"] for row in result.manifest["initial_rows"]] == [
        "BEST",
        "CANSLIM",
        "IPO",
    ]

from __future__ import annotations

import json
from datetime import date

from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsSymbolDetailResponse,
)
from app.services.static_options_artifact_selector import StaticOptionsArtifactSelector
from app.services.static_options_contract import validate_static_options_artifact
from app.services.static_options_exporter import StaticOptionsExporter

from tests.unit.test_static_options_exporter import _item, _Queries, _run


def test_exported_static_contracts_equal_the_live_read_models(tmp_path):
    run = _run(_item("AAPL"), _item("MSFT", rank=2))
    queries = _Queries(run)
    options_dir = tmp_path / "fresh" / "options"

    manifest = StaticOptionsExporter(queries).export(
        options_dir,
        generated_at="2026-09-04T22:00:00Z",
    )

    live_command = OptionsCommandCenterResponse.from_run(run).model_dump(mode="json")
    static_command = json.loads((options_dir / "command-center.json").read_text())
    assert static_command == live_command
    for symbol, entry in manifest["symbols"].items():
        live_detail = OptionsSymbolDetailResponse.from_result(
            queries.get_published_symbol_detail(symbol, "US")
        ).model_dump(mode="json")
        static_detail = json.loads((tmp_path / "fresh" / entry["path"]).read_text())
        assert static_detail == live_detail
    validate_static_options_artifact(options_dir)


def test_fresh_equity_uses_the_prior_valid_options_bundle_as_explicitly_stale(tmp_path):
    prior = tmp_path / "prior" / "options"
    output = tmp_path / "site" / "options"
    StaticOptionsExporter(_Queries(_run(_item("AAPL")))).export(
        prior,
        generated_at="2026-09-04T22:00:00Z",
    )

    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=None,
        fallback_options_dir=prior,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 5),
    )

    assert selected["source_feature_run_id"] == 33
    assert selected["stale_relative_to_equity"] is True
    assert json.loads((output / "command-center.json").read_text())["stale"] is True

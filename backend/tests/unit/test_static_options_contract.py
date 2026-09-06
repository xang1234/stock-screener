from __future__ import annotations

import json

import pytest
from app.services.static_options_contract import (
    StaticOptionsArtifactError,
    validate_static_options_artifact,
)
from app.services.static_options_exporter import StaticOptionsExporter

from .test_static_options_exporter import _item, _Queries, _run


def test_contract_rejects_duplicate_strikes_and_non_finite_values(tmp_path):
    destination = tmp_path / "options"
    manifest = StaticOptionsExporter(_Queries(_run(_item("AAPL")))).export(
        destination,
        generated_at="2026-09-04T22:00:00Z",
    )
    detail_path = tmp_path / manifest["symbols"]["AAPL"]["path"]
    detail = json.loads(detail_path.read_text())
    detail["strike_points"].append(dict(detail["strike_points"][0]))
    detail_path.write_text(json.dumps(detail))

    with pytest.raises(StaticOptionsArtifactError, match="duplicate strike"):
        validate_static_options_artifact(destination)

    detail["strike_points"].pop()
    detail["item"]["spot_price"] = float("inf")
    detail_path.write_text(json.dumps(detail))
    with pytest.raises(StaticOptionsArtifactError, match="non-finite"):
        validate_static_options_artifact(destination)


def test_contract_rejects_run_mismatch_and_unsafe_symbol_path(tmp_path):
    destination = tmp_path / "options"
    manifest = StaticOptionsExporter(_Queries(_run(_item("AAPL")))).export(
        destination,
        generated_at="2026-09-04T22:00:00Z",
    )
    command_path = destination / "command-center.json"
    command = json.loads(command_path.read_text())
    command["run_id"] = 99
    command_path.write_text(json.dumps(command))
    with pytest.raises(StaticOptionsArtifactError, match="run identity mismatch"):
        validate_static_options_artifact(destination)

    command["run_id"] = manifest["published_run_id"]
    command_path.write_text(json.dumps(command))
    manifest["symbols"]["AAPL"]["path"] = "options/../secrets.json"
    (destination / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(StaticOptionsArtifactError, match="unsafe"):
        validate_static_options_artifact(destination)

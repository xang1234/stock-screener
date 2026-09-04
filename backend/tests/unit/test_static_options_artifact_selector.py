from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from app.services.static_options_artifact_selector import StaticOptionsArtifactSelector
from app.services.static_options_exporter import StaticOptionsExporter

from .test_static_options_exporter import _item, _Queries, _run


def _export(
    path,
    *,
    source_run_id=33,
    source_as_of_date=date(2026, 9, 4),
    published_run_id=7,
):
    run = _run(_item("AAPL"))
    run.source_feature_run_id = source_run_id
    run.as_of_date = source_as_of_date
    run.id = published_run_id
    return StaticOptionsExporter(_Queries(run)).export(
        path,
        generated_at="2026-09-04T22:00:00Z",
    )


def test_selector_prefers_current_artifact_matching_fresh_equity(tmp_path):
    current = tmp_path / "current" / "options"
    fallback = tmp_path / "fallback" / "options"
    output = tmp_path / "output" / "options"
    _export(current, source_run_id=44)
    _export(fallback, source_run_id=33)
    unrelated = output.parent / ".options-selected-previous"
    unrelated.mkdir(parents=True)
    (unrelated / "operator.txt").write_text("keep")

    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=current,
        fallback_options_dir=fallback,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 4),
    )

    assert selected is not None
    assert selected["source_feature_run_id"] == 44
    assert selected["stale_relative_to_equity"] is False
    assert (unrelated / "operator.txt").read_text() == "keep"


def test_selector_uses_compatible_last_good_and_marks_every_file_stale(tmp_path):
    current = tmp_path / "current" / "options"
    fallback = tmp_path / "fallback" / "options"
    output = tmp_path / "output" / "options"
    _export(current, source_run_id=22)
    fallback_manifest = _export(fallback, source_run_id=33)

    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=current,
        fallback_options_dir=fallback,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 5),
    )

    assert selected is not None
    assert selected["published_run_id"] == fallback_manifest["published_run_id"]
    assert selected["source_feature_run_id"] == 33
    assert selected["source_as_of_date"] == "2026-09-04"
    assert selected["stale_relative_to_equity"] is True
    assert "stale_relative_to_equity" in selected["reason_codes"]
    command = json.loads((output / "command-center.json").read_text())
    assert command["stale"] is True
    detail_path = tmp_path / "output" / selected["symbols"]["AAPL"]["path"]
    assert json.loads(detail_path.read_text())["stale"] is True


def test_selector_uses_newest_valid_stale_artifact(tmp_path):
    current = tmp_path / "current" / "options"
    fallback = tmp_path / "fallback" / "options"
    output = tmp_path / "output" / "options"
    current_manifest = _export(
        current,
        source_run_id=43,
        source_as_of_date=date(2026, 9, 4),
        published_run_id=8,
    )
    _export(
        fallback,
        source_run_id=42,
        source_as_of_date=date(2026, 9, 3),
        published_run_id=9,
    )

    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=current,
        fallback_options_dir=fallback,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 5),
    )

    assert selected is not None
    assert selected["published_run_id"] == current_manifest["published_run_id"]
    assert selected["source_as_of_date"] == "2026-09-04"


def test_selector_uses_newer_run_as_stale_artifact_date_tiebreaker(tmp_path):
    current = tmp_path / "current" / "options"
    fallback = tmp_path / "fallback" / "options"
    output = tmp_path / "output" / "options"
    current_manifest = _export(current, source_run_id=43, published_run_id=9)
    _export(fallback, source_run_id=42, published_run_id=8)

    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=current,
        fallback_options_dir=fallback,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 5),
    )

    assert selected is not None
    assert selected["published_run_id"] == current_manifest["published_run_id"]


def test_selector_absence_does_not_create_an_options_directory(tmp_path):
    output = tmp_path / "output" / "options"
    selected = StaticOptionsArtifactSelector().select(
        current_options_dir=None,
        fallback_options_dir=None,
        output_options_dir=output,
        equity_feature_run_id=44,
        equity_as_of_date=date(2026, 9, 5),
    )
    assert selected is None
    assert not output.exists()


def test_combine_mode_selects_options_independently_and_advertises_page(
    monkeypatch,
    tmp_path,
):
    from app.services import static_site_export_service as module

    current = tmp_path / "current-options" / "options"
    output = tmp_path / "output"
    _export(current, source_run_id=44)
    us_entry = {
        "market": "US",
        "feature_run_id": 44,
        "as_of_date": "2026-09-04",
        "features": {"scan": True},
        "pages": {},
        "assets": {},
    }
    manifest = {
        "default_market": "US",
        "features": {"scan": True},
        "pages": {},
        "assets": {},
        "markets": {"US": us_entry},
    }

    class _Combiner:
        def __init__(self, **_kwargs):
            pass

        def combine(self, **_kwargs):
            output.mkdir(parents=True)
            return SimpleNamespace(
                output_dir=output,
                generated_at="2026-09-04T22:00:00Z",
                as_of_date="2026-09-04",
                warnings=(),
                manifest=manifest,
            )

    monkeypatch.setattr(module, "StaticArtifactCombiner", _Combiner)

    result = module.StaticSiteExportService.combine_market_artifacts(
        tmp_path / "market-artifacts",
        output,
        options_artifacts_dir=current,
    )

    assert result.manifest["features"]["options"] is True
    assert result.manifest["pages"]["options"] == {"path": "options/manifest.json"}
    assert (output / "options" / "manifest.json").is_file()


def test_fallback_and_validation_scripts_recognize_nested_options_artifact(tmp_path):
    from app.scripts.download_static_market_fallbacks import (
        downloaded_options_as_of_date,
        find_options_artifact_dir,
    )
    from app.scripts.validate_static_market_artifacts import (
        validate_optional_options_artifacts,
    )

    nested = tmp_path / "static-options-US" / "options"
    _export(nested, source_run_id=44)

    assert find_options_artifact_dir(tmp_path) == nested
    assert downloaded_options_as_of_date(tmp_path) == date(2026, 9, 4)
    assert validate_optional_options_artifacts(tmp_path, None)[
        "source_feature_run_id"
    ] == 44

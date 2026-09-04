from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.services.static_options_section import StaticOptionsSection


class _Selector:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def select(self, **kwargs):
        self.calls.append(kwargs)
        output = Path(kwargs["output_options_dir"])
        output.mkdir(parents=True, exist_ok=True)
        return {"schema_version": "static-options-v1"}


def _entry() -> dict[str, object]:
    return {
        "feature_run_id": 42,
        "as_of_date": "2026-09-03",
        "features": {},
        "pages": {},
        "assets": {},
    }


def test_combined_section_advertises_selected_options_in_both_manifests(
    tmp_path: Path,
) -> None:
    selector = _Selector()
    us_entry = _entry()
    manifest = {
        "default_market": "US",
        "markets": {"US": us_entry},
        "features": {},
        "pages": {},
        "assets": {},
    }
    metadata_path = tmp_path / "markets" / "us" / "manifest.market.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({"entry": _entry()}), encoding="utf-8")

    result = StaticOptionsSection(
        enabled=True,
        selector=selector,
    ).compose_combined(
        output_dir=tmp_path,
        manifest=manifest,
        current_options_dir=tmp_path / "current-options",
        fallback_options_dir=None,
        market_metadata_path=metadata_path,
    )

    assert result.selected is True
    assert result.warnings == ()
    assert us_entry["features"] == {"options": True}
    assert manifest["features"] == {"options": True}
    assert manifest["pages"] == {"options": {"path": "options/manifest.json"}}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["entry"]["assets"] == {"options": {"path": "options/manifest.json"}}
    assert selector.calls[0]["equity_feature_run_id"] == 42
    assert selector.calls[0]["equity_as_of_date"] == date(2026, 9, 3)


def test_static_site_exporter_remains_decomposed() -> None:
    service_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "services"
        / "static_site_export_service.py"
    )

    assert len(service_path.read_text(encoding="utf-8").splitlines()) < 1000

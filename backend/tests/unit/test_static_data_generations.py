"""Static data moves under an immutable per-publish generation directory (#504)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.static_data_generations import (
    StaticDataGenerationError,
    relocate_into_generation,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _site(root: Path, *, generated_at: str = "2026-10-08T05:47:12Z") -> None:
    _write(
        root / "manifest.json",
        {
            "generated_at": generated_at,
            "markets": {"US": {"pages": {"scan": {"path": "markets/us/scan/manifest.json"}}}},
        },
    )
    _write(root / "markets/us/scan/manifest.json", {"chunks": [{"path": "markets/us/scan/chunk-1.json"}]})
    _write(root / "markets/us/scan/chunk-1.json", {"rows": []})
    _write(root / "options/manifest.json", {"command_center_path": "options/command_center.json"})


def _files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def test_moves_everything_but_the_root_manifest_into_the_generation(tmp_path):
    _site(tmp_path)
    before = _files(tmp_path) - {"manifest.json"}

    generation = relocate_into_generation(tmp_path)

    assert generation.startswith("20261008T054712Z-")
    data_root = tmp_path / "g" / generation
    assert _files(data_root) == before
    assert {path.name for path in tmp_path.iterdir()} == {"manifest.json", "g"}

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == generation
    assert manifest["data_root"] == f"g/{generation}/"
    # Paths stay root-relative; the frontend resolves them under data_root.
    assert manifest["markets"]["US"]["pages"]["scan"]["path"] == "markets/us/scan/manifest.json"


def test_generation_changes_with_the_manifest_content(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    _site(first)
    _site(second, generated_at="2026-10-08T05:47:13Z")

    assert relocate_into_generation(first) != relocate_into_generation(second)


def test_refuses_an_output_without_a_manifest(tmp_path):
    _write(tmp_path / "markets/us/home.json", {})

    with pytest.raises(StaticDataGenerationError):
        relocate_into_generation(tmp_path)


def test_refuses_an_already_relocated_output(tmp_path):
    _site(tmp_path)
    relocate_into_generation(tmp_path)

    with pytest.raises(StaticDataGenerationError):
        relocate_into_generation(tmp_path)

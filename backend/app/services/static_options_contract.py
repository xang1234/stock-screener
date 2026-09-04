"""Validation contract for an atomic static Options Command Center bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsSymbolDetailResponse,
)
from app.use_cases.options_analytics import (
    OPTIONS_ANALYTICS_CALCULATION_VERSION,
    OPTIONS_ANALYTICS_SCHEMA_VERSION,
)

STATIC_OPTIONS_SCHEMA_VERSION = "static-options-v1"


class StaticOptionsArtifactError(ValueError):
    """Raised when a static options bundle is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticOptionsArtifactError(
            f"invalid options artifact file: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise StaticOptionsArtifactError(f"options artifact must be an object: {path}")
    _require_finite(payload, location=path.as_posix())
    return payload


def _require_finite(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StaticOptionsArtifactError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, location=f"{location}[{index}]")


def _artifact_path(options_dir: Path, advertised_path: str) -> Path:
    relative = Path(str(advertised_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise StaticOptionsArtifactError(
            f"unsafe options artifact path: {advertised_path}"
        )
    if not relative.parts or relative.parts[0] != "options":
        raise StaticOptionsArtifactError(
            f"unsafe options artifact path: {advertised_path}"
        )
    root = options_dir.resolve()
    resolved = (root / Path(*relative.parts[1:])).resolve()
    if resolved != root and root not in resolved.parents:
        raise StaticOptionsArtifactError(
            f"unsafe options artifact path: {advertised_path}"
        )
    return resolved


def _same_run(payload: Any, manifest: dict[str, Any], *, location: str) -> None:
    expected = {
        "schema_version": manifest["data_schema_version"],
        "calculation_version": manifest["calculation_version"],
        "run_id": manifest["published_run_id"],
        "source_feature_run_id": manifest["source_feature_run_id"],
        "source_as_of_date": manifest["source_as_of_date"],
        "market": manifest["market"],
        "provider": manifest["provider"],
        "latest_observation_at": manifest["latest_observation_at"],
        "coverage": manifest["coverage"],
        "stale": manifest["stale"],
    }
    actual = payload.model_dump(mode="json")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise StaticOptionsArtifactError(
                f"run identity mismatch for {location}: {key}"
            )


def validate_static_options_artifact(
    options_dir: Path,
    *,
    required_calculation_version: str = OPTIONS_ANALYTICS_CALCULATION_VERSION,
) -> dict[str, Any]:
    """Validate a complete options directory and return its manifest."""

    options_dir = Path(options_dir)
    manifest = _load_json(options_dir / "manifest.json")
    required_fields = {
        "schema_version",
        "data_schema_version",
        "calculation_version",
        "published_run_id",
        "source_feature_run_id",
        "source_as_of_date",
        "market",
        "provider",
        "generated_at",
        "latest_observation_at",
        "coverage",
        "stale",
        "stale_relative_to_equity",
        "reason_codes",
        "command_center_path",
        "symbols",
    }
    missing = required_fields.difference(manifest)
    if missing:
        raise StaticOptionsArtifactError(
            f"options manifest missing fields: {', '.join(sorted(missing))}"
        )
    if manifest["schema_version"] != STATIC_OPTIONS_SCHEMA_VERSION:
        raise StaticOptionsArtifactError("unsupported static options schema version")
    if manifest["data_schema_version"] != OPTIONS_ANALYTICS_SCHEMA_VERSION:
        raise StaticOptionsArtifactError("unsupported options data schema version")
    if manifest["calculation_version"] != required_calculation_version:
        raise StaticOptionsArtifactError("incompatible options calculation version")
    if manifest["market"] != "US":
        raise StaticOptionsArtifactError("static options artifact must be US scoped")
    if not isinstance(manifest["published_run_id"], int):
        raise StaticOptionsArtifactError("invalid published options run identity")
    coverage = manifest["coverage"]
    if not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1:
        raise StaticOptionsArtifactError("invalid options coverage")
    if not isinstance(manifest["symbols"], dict):
        raise StaticOptionsArtifactError("invalid options symbol map")
    if len(manifest["symbols"]) > 80:
        raise StaticOptionsArtifactError("options current cohort exceeds 80 symbols")
    if manifest["stale_relative_to_equity"] and (
        not manifest["stale"]
        or "stale_relative_to_equity" not in manifest["reason_codes"]
    ):
        raise StaticOptionsArtifactError("invalid stale options metadata")

    command_path = _artifact_path(options_dir, manifest["command_center_path"])
    try:
        command = OptionsCommandCenterResponse.model_validate(_load_json(command_path))
    except ValidationError as exc:
        raise StaticOptionsArtifactError(
            "invalid options command-center contract"
        ) from exc
    _same_run(command, manifest, location="command-center")
    command_symbols = [item.symbol for item in command.items]
    if len(command_symbols) != len(set(command_symbols)):
        raise StaticOptionsArtifactError("duplicate symbol in options command center")
    if command.current_count != len(command_symbols):
        raise StaticOptionsArtifactError(
            "options current count does not match summaries"
        )
    if set(command_symbols) != set(manifest["symbols"]):
        raise StaticOptionsArtifactError(
            "options symbol map does not cover current cohort"
        )

    advertised_paths: set[str] = set()
    for item in command.items:
        if not item.source_badges:
            raise StaticOptionsArtifactError(
                f"continuity-only symbol exported in command center: {item.symbol}"
            )
        entry = manifest["symbols"][item.symbol]
        if not isinstance(entry, dict) or not {"key", "path"}.issubset(entry):
            raise StaticOptionsArtifactError(f"missing detail path for {item.symbol}")
        if entry["path"] in advertised_paths:
            raise StaticOptionsArtifactError("duplicate options symbol detail path")
        advertised_paths.add(entry["path"])
        detail_path = _artifact_path(options_dir, entry["path"])
        if Path(entry["path"]).stem != entry["key"]:
            raise StaticOptionsArtifactError(
                f"symbol key/path mismatch for {item.symbol}"
            )
        try:
            detail = OptionsSymbolDetailResponse.model_validate(_load_json(detail_path))
        except ValidationError as exc:
            raise StaticOptionsArtifactError(
                f"invalid options symbol contract: {item.symbol}"
            ) from exc
        _same_run(detail, manifest, location=item.symbol)
        if detail.item.symbol != item.symbol:
            raise StaticOptionsArtifactError(
                f"symbol detail mismatch for {item.symbol}"
            )
        strikes = [point.strike for point in detail.strike_points]
        if len(strikes) != len(set(strikes)):
            raise StaticOptionsArtifactError(f"duplicate strike for {item.symbol}")

    return manifest


__all__ = [
    "STATIC_OPTIONS_SCHEMA_VERSION",
    "StaticOptionsArtifactError",
    "validate_static_options_artifact",
]

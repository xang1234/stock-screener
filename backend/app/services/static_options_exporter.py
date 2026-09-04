"""Atomic serializer for the published Options Command Center read models."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.infra.serialization import json_safe
from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsSymbolDetailResponse,
)
from app.services.static_options_contract import (
    STATIC_OPTIONS_SCHEMA_VERSION,
    validate_static_options_artifact,
)


class StaticOptionsUnavailable(RuntimeError):
    """Raised when no complete Published Options Run can be exported."""


def url_safe_symbol_key(symbol: str) -> str:
    canonical = symbol.strip().upper().encode("utf-8")
    return base64.urlsafe_b64encode(canonical).decode("ascii").rstrip("=")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


class StaticOptionsExporter:
    def __init__(self, queries: Any) -> None:
        self._queries = queries

    def export(self, options_dir: Path, *, generated_at: str) -> dict[str, Any]:
        destination = Path(options_dir)
        if destination.name != "options":
            raise ValueError("Static options destination must be named 'options'")
        run = self._queries.get_published_command_center("US")
        if run is None:
            raise StaticOptionsUnavailable("No Published Options Run is available")
        stale = self._queries.is_stale(run, "US")
        command = OptionsCommandCenterResponse.from_run(run, stale=stale)

        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=".options-stage-", dir=str(destination.parent))
        )
        backup = Path(
            tempfile.mkdtemp(prefix=".options-backup-", dir=str(destination.parent))
        )
        backup.rmdir()
        try:
            symbol_map: dict[str, dict[str, str]] = {}
            for item in command.items:
                result = self._queries.get_published_symbol_detail(item.symbol, "US")
                if result is None or result.run.id != run.id:
                    raise StaticOptionsUnavailable(
                        f"Published options detail unavailable for {item.symbol}"
                    )
                key = url_safe_symbol_key(item.symbol)
                relative_path = f"options/symbols/{key}.json"
                symbol_map[item.symbol] = {"key": key, "path": relative_path}
                detail = OptionsSymbolDetailResponse.from_result(result, stale=stale)
                _write_json(
                    stage / "symbols" / f"{key}.json", detail.model_dump(mode="json")
                )

            command_payload = command.model_dump(mode="json")
            _write_json(stage / "command-center.json", command_payload)
            manifest = {
                "schema_version": STATIC_OPTIONS_SCHEMA_VERSION,
                "data_schema_version": command.schema_version,
                "calculation_version": command.calculation_version,
                "published_run_id": command.run_id,
                "source_feature_run_id": command.source_feature_run_id,
                "source_as_of_date": command.source_as_of_date.isoformat(),
                "market": command.market,
                "provider": command.provider,
                "generated_at": generated_at,
                "latest_observation_at": command_payload["latest_observation_at"],
                "coverage": command.coverage,
                "stale": stale,
                "stale_relative_to_equity": stale,
                "reason_codes": (["stale_relative_to_equity"] if stale else []),
                "command_center_path": "options/command-center.json",
                "symbols": symbol_map,
            }
            _write_json(stage / "manifest.json", manifest)
            validate_static_options_artifact(stage)

            if destination.exists():
                destination.rename(backup)
            try:
                stage.rename(destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            if backup.exists():
                shutil.rmtree(backup)


__all__ = ["StaticOptionsExporter", "StaticOptionsUnavailable", "url_safe_symbol_key"]

"""Select and promote current or last-good static options artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.infra.serialization import json_safe
from app.services.atomic_directory_publisher import AtomicDirectoryPublisher
from app.services.static_options_contract import (
    StaticOptionsArtifactError,
    validate_static_options_artifact,
)


def _validated(path: Path | None) -> tuple[Path, dict[str, Any]] | None:
    if path is None or not Path(path).is_dir():
        return None
    try:
        return Path(path), validate_static_options_artifact(Path(path))
    except StaticOptionsArtifactError:
        return None


def _stale_order_key(artifact: tuple[Path, dict[str, Any]]) -> tuple[date, datetime]:
    manifest = artifact[1]
    generated_at = datetime.fromisoformat(manifest["generated_at"])
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return (
        date.fromisoformat(manifest["source_as_of_date"]),
        generated_at.astimezone(UTC),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _mark_stale(options_dir: Path, *, equity_run_id: int, equity_date: date) -> None:
    manifest_path = options_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stale"] = True
    manifest["stale_relative_to_equity"] = True
    manifest["equity_feature_run_id"] = equity_run_id
    manifest["equity_as_of_date"] = equity_date.isoformat()
    reasons = list(manifest.get("reason_codes") or [])
    if "stale_relative_to_equity" not in reasons:
        reasons.append("stale_relative_to_equity")
    manifest["reason_codes"] = reasons
    _write_json(manifest_path, manifest)

    payload_paths = [options_dir / "command-center.json"]
    payload_paths.extend(
        options_dir / Path(*Path(entry["path"]).parts[1:])
        for entry in manifest["symbols"].values()
    )
    for path in payload_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["stale"] = True
        reason_codes = list(payload.get("reason_codes") or [])
        if "stale_relative_to_equity" not in reason_codes:
            reason_codes.append("stale_relative_to_equity")
        payload["reason_codes"] = reason_codes
        _write_json(path, payload)


class StaticOptionsArtifactSelector:
    def __init__(
        self,
        *,
        publisher: AtomicDirectoryPublisher | None = None,
    ) -> None:
        self._publisher = publisher or AtomicDirectoryPublisher()

    def select(
        self,
        *,
        current_options_dir: Path | None,
        fallback_options_dir: Path | None,
        output_options_dir: Path,
        equity_feature_run_id: int,
        equity_as_of_date: date,
    ) -> dict[str, Any] | None:
        current = _validated(current_options_dir)
        fallback = _validated(fallback_options_dir)
        selected: tuple[Path, dict[str, Any]] | None = None
        stale = False
        if current is not None:
            current_manifest = current[1]
            if (
                current_manifest["source_feature_run_id"] == equity_feature_run_id
                and current_manifest["source_as_of_date"]
                == equity_as_of_date.isoformat()
                and not current_manifest["stale"]
            ):
                selected = current
        if selected is None:
            stale_candidates = tuple(
                artifact for artifact in (fallback, current) if artifact is not None
            )
            selected = (
                max(
                    stale_candidates,
                    key=lambda artifact: (
                        *_stale_order_key(artifact),
                        artifact is current,
                    ),
                )
                if stale_candidates
                else None
            )
            stale = selected is not None
        if selected is None:
            return None

        output = Path(output_options_dir)

        def populate(stage: Path) -> None:
            shutil.copytree(selected[0], stage, dirs_exist_ok=True)
            if stale:
                _mark_stale(
                    stage,
                    equity_run_id=equity_feature_run_id,
                    equity_date=equity_as_of_date,
                )
            else:
                manifest_path = stage / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["equity_feature_run_id"] = equity_feature_run_id
                manifest["equity_as_of_date"] = equity_as_of_date.isoformat()
                _write_json(manifest_path, manifest)

        self._publisher.publish(
            output,
            populate,
            validate=validate_static_options_artifact,
        )
        return validate_static_options_artifact(output)


__all__ = ["StaticOptionsArtifactSelector"]

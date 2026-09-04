"""Options-specific composition for direct and combined static exports."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.infra.serialization import json_safe
from app.services.static_options_artifact_selector import StaticOptionsArtifactSelector
from app.services.static_options_contract import StaticOptionsArtifactError
from app.services.static_options_exporter import (
    StaticOptionsExporter,
    StaticOptionsUnavailable,
)

OptionsExporterFactory = Callable[[Session], StaticOptionsExporter]


@dataclass(frozen=True)
class StaticOptionsSectionResult:
    selected: bool
    warnings: tuple[str, ...] = ()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _default_exporter_factory(db: Session) -> StaticOptionsExporter:
    from app.wiring.bootstrap import get_options_analytics_queries

    return StaticOptionsExporter(get_options_analytics_queries(db))


class StaticOptionsSection:
    """Own options export, fallback selection, and manifest contribution."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        exporter_factory: OptionsExporterFactory | None = None,
        selector: StaticOptionsArtifactSelector | None = None,
        json_writer: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> None:
        self._enabled = (
            settings.options_analytics_enabled if enabled is None else enabled
        )
        self._exporter_factory = exporter_factory or _default_exporter_factory
        self._selector = selector or StaticOptionsArtifactSelector()
        self._write_json = json_writer or _write_json

    def compose_live(
        self,
        *,
        db: Session,
        output_dir: Path,
        generated_at: str,
        equity_entry: dict[str, Any],
        fallback_options_dir: Path | None,
    ) -> StaticOptionsSectionResult:
        if not self._enabled:
            return StaticOptionsSectionResult(selected=False)

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        candidate_root = Path(
            tempfile.mkdtemp(prefix=".options-current-", dir=str(root))
        )
        candidate_options = candidate_root / "options"
        current: Path | None = None
        warnings: list[str] = []
        try:
            try:
                self._exporter_factory(db).export(
                    candidate_options,
                    generated_at=generated_at,
                )
                current = candidate_options
            except (StaticOptionsUnavailable, StaticOptionsArtifactError) as exc:
                warnings.append(f"Current US options artifact unavailable: {exc}")

            selected = self._select(
                current_options_dir=current,
                fallback_options_dir=fallback_options_dir,
                output_dir=root,
                equity_entry=equity_entry,
            )
        finally:
            if candidate_root.exists():
                shutil.rmtree(candidate_root)

        if not selected:
            warnings.append("Static US options data is unavailable")
            return StaticOptionsSectionResult(False, tuple(warnings))
        self._advertise(equity_entry)
        return StaticOptionsSectionResult(True, tuple(warnings))

    def compose_combined(
        self,
        *,
        output_dir: Path,
        manifest: dict[str, Any],
        current_options_dir: Path | None,
        fallback_options_dir: Path | None,
        market_metadata_path: Path | None,
    ) -> StaticOptionsSectionResult:
        if not self._enabled:
            return StaticOptionsSectionResult(selected=False)
        us_entry = (manifest.get("markets") or {}).get("US")
        if us_entry is None or us_entry.get("feature_run_id") is None:
            return StaticOptionsSectionResult(selected=False)

        selected = self._select(
            current_options_dir=current_options_dir,
            fallback_options_dir=fallback_options_dir,
            output_dir=Path(output_dir),
            equity_entry=us_entry,
        )
        if not selected:
            warnings = (
                ("No compatible static US options artifact was available",)
                if current_options_dir is not None or fallback_options_dir is not None
                else ()
            )
            return StaticOptionsSectionResult(False, warnings)

        self._advertise(us_entry)
        if manifest.get("default_market") == "US":
            for section in ("features", "pages", "assets"):
                manifest[section] = dict(us_entry[section])
        self._write_json(Path(output_dir) / "manifest.json", manifest)

        if market_metadata_path is not None and Path(market_metadata_path).is_file():
            metadata = json.loads(
                Path(market_metadata_path).read_text(encoding="utf-8")
            )
            metadata["entry"] = us_entry
            self._write_json(Path(market_metadata_path), metadata)
        return StaticOptionsSectionResult(selected=True)

    def _select(
        self,
        *,
        current_options_dir: Path | None,
        fallback_options_dir: Path | None,
        output_dir: Path,
        equity_entry: dict[str, Any],
    ) -> bool:
        selected = self._selector.select(
            current_options_dir=current_options_dir,
            fallback_options_dir=fallback_options_dir,
            output_options_dir=output_dir / "options",
            equity_feature_run_id=int(equity_entry["feature_run_id"]),
            equity_as_of_date=date.fromisoformat(equity_entry["as_of_date"]),
        )
        return selected is not None

    @staticmethod
    def _advertise(entry: dict[str, Any]) -> None:
        entry.setdefault("features", {})["options"] = True
        for section in ("pages", "assets"):
            entry.setdefault(section, {})["options"] = {"path": "options/manifest.json"}


__all__ = ["StaticOptionsSection", "StaticOptionsSectionResult"]

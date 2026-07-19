from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from app.services.static_site_errors import NoPublishedStaticMarketArtifact


@dataclass(frozen=True)
class StaticArtifactCombineResult:
    output_dir: Path
    generated_at: str
    as_of_date: str
    warnings: tuple[str, ...]
    manifest: dict[str, Any]


class StaticArtifactFormulaError(RuntimeError):
    """A Market artifact does not match its required RS formula contract."""


class StaticArtifactCombiner:
    def __init__(
        self,
        *,
        schema_version: str,
        supported_markets: tuple[str, ...],
        default_market: str,
        metadata_filename: str = "manifest.market.json",
    ) -> None:
        self._schema_version = schema_version
        self._supported_markets = tuple(supported_markets)
        self._default_market = default_market
        self._metadata_filename = metadata_filename

    def combine(
        self,
        *,
        artifacts_dir: Path,
        fallback_artifacts_dir: Path | None,
        output_dir: Path,
        required_formula_by_market: Mapping[str, str],
        clean: bool,
    ) -> StaticArtifactCombineResult:
        required = {
            str(market).strip().upper(): str(formula).strip()
            for market, formula in required_formula_by_market.items()
        }
        current = self._discover(
            Path(artifacts_dir),
            source_label="current",
            required=required,
        )
        fallback = (
            self._discover(
                Path(fallback_artifacts_dir),
                source_label="fallback",
                required=required,
            )
            if fallback_artifacts_dir is not None
            else {}
        )
        selected = dict(current)
        for market, artifact in fallback.items():
            selected.setdefault(market, artifact)

        if required:
            missing = sorted(market for market in required if market not in selected)
            if missing:
                raise NoPublishedStaticMarketArtifact(
                    "No published compatible static artifact is available for required "
                    f"Markets: {', '.join(missing)}.",
                    markets=tuple(missing),
                )
        elif not selected:
            raise RuntimeError(
                "No market artifacts are available to combine into a static-site bundle"
            )

        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []
        entries: dict[str, dict[str, Any]] = {}
        for market, artifact in selected.items():
            entries[market] = artifact["entry"]
            warnings.extend(str(item) for item in artifact["metadata"].get("warnings", []))
            if artifact["source_label"] == "fallback":
                warnings.append(
                    f"{market} reused from a previous static-site market artifact "
                    "because the current run produced no artifact."
                )
        if not required:
            warnings.extend(
                f"Static export market {market} was omitted from the combined bundle "
                "because no artifact was produced."
                for market in self._supported_markets
                if market not in entries
            )
        manifest = self._build_manifest(
            market_entries=entries,
            generated_at=generated_at,
            warnings=warnings,
        )
        self._publish(
            selected=selected,
            output_dir=Path(output_dir),
            manifest=manifest,
            clean=clean,
        )
        return StaticArtifactCombineResult(
            output_dir=Path(output_dir),
            generated_at=generated_at,
            as_of_date=manifest["as_of_date"],
            warnings=tuple(warnings),
            manifest=manifest,
        )

    def _discover(
        self,
        root: Path,
        *,
        source_label: str,
        required: Mapping[str, str],
    ) -> dict[str, dict[str, Any]]:
        discovered: dict[str, dict[str, Any]] = {}
        paths = sorted(root.rglob(self._metadata_filename)) if root.exists() else []
        for metadata_path in paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            market = str(metadata.get("market") or "").strip().upper()
            if not market:
                raise RuntimeError(f"Invalid Market metadata at {metadata_path}")
            if market in discovered:
                raise RuntimeError(f"Duplicate {source_label} artifact for {market}")
            if metadata.get("schema_version") != self._schema_version:
                raise RuntimeError(
                    f"{market} {source_label} artifact uses schema_version "
                    f"{metadata.get('schema_version')!r}; expected {self._schema_version!r}"
                )
            market_dir = metadata_path.parent
            entry = metadata.get("entry")
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"{market} {source_label} metadata has no Market entry"
                )
            expected = required.get(market)
            if expected is not None:
                entry = self._validate_formula(
                    market=market,
                    source_label=source_label,
                    metadata=metadata,
                    market_dir=market_dir,
                    expected_formula=expected,
                )
            self._validate_advertised_assets(
                market=market,
                source_label=source_label,
                entry=entry,
                market_dir=market_dir,
            )
            discovered[market] = {
                "entry": entry,
                "metadata": metadata,
                "market_dir": market_dir,
                "source_label": source_label,
            }
        return discovered

    @staticmethod
    def _validate_advertised_assets(
        *, market: str, source_label: str, entry: dict, market_dir: Path
    ) -> None:
        features = entry.get("features") if isinstance(entry.get("features"), dict) else {}
        for feature, filename in (("groups", "groups.json"), ("rrg", "groups_rrg.json")):
            if features.get(feature) and not (market_dir / filename).is_file():
                raise StaticArtifactFormulaError(
                    f"{market} {source_label} artifact advertises "
                    f"{feature.upper()} but {filename} is absent"
                )

    @classmethod
    def _validate_formula(
        cls,
        *,
        market: str,
        source_label: str,
        metadata: dict,
        market_dir: Path,
        expected_formula: str,
    ) -> dict:
        entry = metadata.get("entry")
        if not isinstance(entry, dict):
            raise RuntimeError(f"{market} {source_label} metadata has no Market entry")
        observed = {"market entry": entry.get("rs_formula_version")}
        features = entry.get("features") if isinstance(entry.get("features"), dict) else {}
        groups_path = market_dir / "groups.json"
        if features.get("groups") and not groups_path.is_file():
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact advertises Groups but groups.json is absent"
            )
        if groups_path.is_file():
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
            if groups.get("available", True):
                observed["Groups"] = groups.get("rs_formula_version")
        rrg_path = market_dir / "groups_rrg.json"
        if features.get("rrg") and not rrg_path.is_file():
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact advertises RRG but groups_rrg.json is absent"
            )
        if rrg_path.is_file():
            rrg = json.loads(rrg_path.read_text(encoding="utf-8"))
            if rrg.get("available", True):
                observed["RRG"] = rrg.get("rs_formula_version")
        mismatches = {
            source: formula
            for source, formula in observed.items()
            if formula != expected_formula
        }
        if mismatches:
            rendered = ", ".join(
                f"{source}={formula!r}"
                for source, formula in sorted(mismatches.items())
            )
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact uses incompatible RS formula: "
                f"{rendered}; expected {expected_formula!r}"
            )
        return entry

    def _build_manifest(
        self,
        *,
        market_entries: dict[str, dict[str, Any]],
        generated_at: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        ordered_markets = [
            market for market in self._supported_markets if market in market_entries
        ]
        ordered_entries = {market: market_entries[market] for market in ordered_markets}
        default_market = (
            self._default_market
            if self._default_market in ordered_entries
            else next(iter(ordered_entries))
        )
        default_entry = ordered_entries[default_market]
        return {
            "schema_version": self._schema_version,
            "generated_at": generated_at,
            "as_of_date": default_entry["as_of_date"],
            "default_market": default_market,
            "supported_markets": ordered_markets,
            "features": dict(default_entry["features"]),
            "pages": dict(default_entry["pages"]),
            "assets": dict(default_entry["assets"]),
            "markets": ordered_entries,
            "warnings": list(warnings),
        }

    @staticmethod
    def _publish(
        *,
        selected: dict[str, dict[str, Any]],
        output_dir: Path,
        manifest: dict[str, Any],
        clean: bool,
    ) -> None:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
        )
        backup = output_dir.parent / f".{output_dir.name}.previous"
        try:
            if not clean and output_dir.exists():
                shutil.copytree(output_dir, stage, dirs_exist_ok=True)
            for market, artifact in selected.items():
                shutil.copytree(
                    artifact["market_dir"],
                    stage / "markets" / market.lower(),
                    dirs_exist_ok=True,
                )
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if backup.exists():
                shutil.rmtree(backup)
            if output_dir.exists():
                output_dir.rename(backup)
            try:
                stage.rename(output_dir)
            except Exception:
                if backup.exists() and not output_dir.exists():
                    backup.rename(output_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

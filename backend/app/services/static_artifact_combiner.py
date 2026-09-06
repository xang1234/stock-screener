from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.services.atomic_directory_publisher import AtomicDirectoryPublisher
from app.services.breadth.types import CURRENT_BREADTH_CALCULATION_REVISION
from app.services.static_breadth_contributor_asset_validator import (
    StaticBreadthContributorAssetError,
    validate_static_breadth_contributor_asset,
)
from app.services.static_market_artifact_contract import (
    STATIC_MARKET_METADATA_FILENAME,
    expected_market_from_static_market_manifest_path,
    read_static_market_manifest,
)
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
        metadata_filename: str = STATIC_MARKET_METADATA_FILENAME,
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
        fallback_required_formula_by_market: Mapping[str, str] | None = None,
        optional_markets: Iterable[str] = (),
        clean: bool,
    ) -> StaticArtifactCombineResult:
        required = {
            str(market).strip().upper(): str(formula).strip()
            for market, formula in required_formula_by_market.items()
        }
        optional = {
            str(market).strip().upper()
            for market in optional_markets
            if str(market).strip()
        }
        fallback_required = (
            required
            if fallback_required_formula_by_market is None
            else {
                str(market).strip().upper(): str(formula).strip()
                for market, formula in fallback_required_formula_by_market.items()
            }
        )
        current = self._discover(
            Path(artifacts_dir),
            source_label="current",
            required={},
        )
        fallback = (
            self._discover(
                Path(fallback_artifacts_dir),
                source_label="fallback",
                required={},
            )
            if fallback_artifacts_dir is not None
            else {}
        )
        selected, fallback_reasons = self._select_artifacts(
            current=current,
            fallback=fallback,
            fallback_required=fallback_required,
        )

        if required:
            missing = sorted(
                market
                for market in required
                if market not in selected and market not in optional
            )
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
        self._validate_selected_formulas(
            selected=selected,
            required=required,
            fallback_required=fallback_required,
        )

        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []
        entries: dict[str, dict[str, Any]] = {}
        for market, artifact in selected.items():
            entries[market] = artifact["entry"]
            warnings.extend(
                str(item) for item in artifact["metadata"].get("warnings", [])
            )
            if artifact["source_label"] == "fallback":
                if fallback_reasons.get(market) == "newer":
                    warnings.append(
                        f"{market} reused from a previous static-site market artifact "
                        "because it is newer than the current artifact."
                    )
                else:
                    warnings.append(
                        f"{market} reused from a previous static-site market artifact "
                        "because the current run produced no artifact."
                    )
        optional_missing = sorted(
            market for market in optional if market not in entries
        )
        warnings.extend(
            f"Static export market {market} was omitted from the combined bundle "
            "because no current or fallback artifact was available."
            for market in optional_missing
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
            omitted_markets=optional_missing,
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

    @classmethod
    def _select_artifacts(
        cls,
        *,
        current: dict[str, dict[str, Any]],
        fallback: dict[str, dict[str, Any]],
        fallback_required: Mapping[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        selected = dict(current)
        fallback_reasons: dict[str, str] = {}
        for market, fallback_artifact in fallback.items():
            if not cls._artifact_matches_breadth_revision(
                market=market,
                artifact=fallback_artifact,
            ):
                continue
            expected_fallback_formula = fallback_required.get(market)
            if (
                expected_fallback_formula is not None
                and not cls._artifact_matches_formula(
                    market=market,
                    artifact=fallback_artifact,
                    expected_formula=expected_fallback_formula,
                )
            ):
                continue
            current_artifact = selected.get(market)
            if current_artifact is None:
                selected[market] = fallback_artifact
                fallback_reasons[market] = "missing"
                continue
            if cls._artifact_is_newer(fallback_artifact, current_artifact):
                selected[market] = fallback_artifact
                fallback_reasons[market] = "newer"
        return selected, fallback_reasons

    @classmethod
    def _artifact_is_newer(
        cls,
        candidate: dict[str, Any],
        incumbent: dict[str, Any],
    ) -> bool:
        candidate_date = cls._artifact_as_of_date(candidate)
        incumbent_date = cls._artifact_as_of_date(incumbent)
        return candidate_date is not None and (
            incumbent_date is None or candidate_date > incumbent_date
        )

    @staticmethod
    def _artifact_as_of_date(artifact: dict[str, Any]) -> date | None:
        entry = artifact.get("entry")
        value = entry.get("as_of_date") if isinstance(entry, dict) else None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text.split("T", 1)[0])
        except ValueError:
            return None

    @classmethod
    def _artifact_matches_formula(
        cls,
        *,
        market: str,
        artifact: dict[str, Any],
        expected_formula: str,
    ) -> bool:
        try:
            cls._validate_formula(
                market=market,
                source_label=artifact["source_label"],
                metadata=artifact["metadata"],
                market_dir=artifact["market_dir"],
                expected_formula=expected_formula,
            )
        except StaticArtifactFormulaError:
            return False
        return True

    @classmethod
    def _artifact_matches_breadth_revision(
        cls,
        *,
        market: str,
        artifact: dict[str, Any],
    ) -> bool:
        try:
            cls._validate_breadth_revision(
                market=market,
                source_label=artifact["source_label"],
                metadata=artifact["metadata"],
                market_dir=artifact["market_dir"],
            )
        except StaticArtifactFormulaError:
            return False
        return True

    @classmethod
    def _validate_selected_formulas(
        cls,
        *,
        selected: dict[str, dict[str, Any]],
        required: Mapping[str, str],
        fallback_required: Mapping[str, str],
    ) -> None:
        for market, artifact in selected.items():
            required_formulas = (
                fallback_required
                if artifact["source_label"] == "fallback"
                else required
            )
            expected = required_formulas.get(market)
            if expected is not None:
                artifact["entry"] = cls._validate_formula(
                    market=market,
                    source_label=artifact["source_label"],
                    metadata=artifact["metadata"],
                    market_dir=artifact["market_dir"],
                    expected_formula=expected,
                )
            artifact["entry"] = cls._validate_breadth_revision(
                market=market,
                source_label=artifact["source_label"],
                metadata=artifact["metadata"],
                market_dir=artifact["market_dir"],
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
            expected_market = expected_market_from_static_market_manifest_path(
                root,
                metadata_path,
            )
            metadata = read_static_market_manifest(
                metadata_path,
                expected_schema_version=self._schema_version,
                expected_market=expected_market,
            )
            market = str(metadata.get("market") or "").strip().upper()
            if market in discovered:
                raise RuntimeError(f"Duplicate {source_label} artifact for {market}")
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
            asset_warnings = self._validate_advertised_assets(
                market=market,
                source_label=source_label,
                entry=entry,
                market_dir=market_dir,
            )
            if asset_warnings:
                metadata["warnings"] = [
                    *metadata.get("warnings", []),
                    *asset_warnings,
                ]
                metadata["entry"] = entry
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
    ) -> list[str]:
        warnings: list[str] = []
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        for feature, filename in (
            ("groups", "groups.json"),
            ("rrg", "groups_rrg.json"),
        ):
            if features.get(feature) and not (market_dir / filename).is_file():
                raise StaticArtifactFormulaError(
                    f"{market} {source_label} artifact advertises "
                    f"{feature.upper()} but {filename} is absent"
                )
        assets = entry.get("assets")
        contributor_asset = (
            assets.get("breadth_contributors") if isinstance(assets, dict) else None
        )
        if contributor_asset is not None:
            try:
                StaticArtifactCombiner._validate_breadth_contributor_asset(
                    market=market,
                    market_dir=market_dir,
                    descriptor=contributor_asset,
                )
            except (
                OSError,
                TypeError,
                ValueError,
                StaticArtifactFormulaError,
                StaticBreadthContributorAssetError,
            ) as exc:
                entry["assets"] = dict(assets)
                entry["assets"].pop("breadth_contributors", None)
                warnings.append(f"{market} breadth contributor asset ignored: {exc}")
        return warnings

    @staticmethod
    def _validate_breadth_contributor_asset(
        *,
        market: str,
        market_dir: Path,
        descriptor: object,
    ) -> None:
        try:
            validate_static_breadth_contributor_asset(
                market=market,
                market_dir=market_dir,
                descriptor=descriptor,
            )
        except StaticBreadthContributorAssetError as exc:
            raise StaticArtifactFormulaError(str(exc)) from exc

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
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        scan_manifest_path = market_dir / "scan" / "manifest.json"
        if features.get("scan") and not scan_manifest_path.is_file():
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact advertises Scan but "
                "scan/manifest.json is absent"
            )
        if scan_manifest_path.is_file():
            scan_manifest = json.loads(scan_manifest_path.read_text(encoding="utf-8"))
            observed["Scan manifest"] = scan_manifest.get("rs_formula_version")
            artifact_root = market_dir.resolve()
            published_market_prefix = Path("markets") / market.lower()
            for chunk_ref in scan_manifest.get("chunks") or []:
                advertised_path = str(chunk_ref.get("path") or "").strip()
                relative_path = Path(advertised_path)
                try:
                    relative_path = relative_path.relative_to(published_market_prefix)
                except ValueError:
                    # Older manifests may already advertise paths relative to
                    # the per-market artifact root.
                    pass
                chunk_path = (artifact_root / relative_path).resolve()
                try:
                    chunk_path.relative_to(artifact_root)
                except ValueError as exc:
                    raise StaticArtifactFormulaError(
                        f"{market} {source_label} Scan chunk path escapes its artifact: "
                        f"{advertised_path!r}"
                    ) from exc
                if not advertised_path or not chunk_path.is_file():
                    raise StaticArtifactFormulaError(
                        f"{market} {source_label} Scan chunk is absent: "
                        f"{advertised_path!r}"
                    )
                chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
                observed[f"Scan chunk {chunk_path.name}"] = chunk.get(
                    "rs_formula_version"
                )
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

    @staticmethod
    def _validate_breadth_revision(
        *,
        market: str,
        source_label: str,
        metadata: dict,
        market_dir: Path,
    ) -> dict:
        entry = metadata.get("entry")
        if not isinstance(entry, dict):
            raise RuntimeError(f"{market} {source_label} metadata has no Market entry")
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        if not features.get("breadth"):
            return entry

        breadth_path = market_dir / "breadth.json"
        if not breadth_path.is_file():
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact advertises breadth but "
                "breadth.json is absent"
            )
        breadth = json.loads(breadth_path.read_text(encoding="utf-8"))
        payload = breadth.get("payload")
        current = payload.get("current") if isinstance(payload, dict) else None
        observed_revision = (
            current.get("calculation_revision") if isinstance(current, dict) else None
        )
        source_revision = str(breadth.get("source_revision") or "")
        expected_marker = f"|breadth-r{CURRENT_BREADTH_CALCULATION_REVISION}"
        if (
            observed_revision != CURRENT_BREADTH_CALCULATION_REVISION
            or not source_revision.endswith(expected_marker)
        ):
            raise StaticArtifactFormulaError(
                f"{market} {source_label} artifact uses incompatible breadth revision: "
                f"revision={observed_revision!r}, source_revision={source_revision!r}; "
                f"expected {CURRENT_BREADTH_CALCULATION_REVISION} and marker "
                f"{expected_marker!r}"
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
        omitted_markets: Iterable[str],
        output_dir: Path,
        manifest: dict[str, Any],
        clean: bool,
    ) -> None:
        def populate(stage: Path) -> None:
            for market in omitted_markets:
                omitted_dir = stage / "markets" / market.lower()
                if omitted_dir.exists() or omitted_dir.is_symlink():
                    shutil.rmtree(omitted_dir)
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

        AtomicDirectoryPublisher().publish(
            output_dir,
            populate,
            clean=clean,
        )

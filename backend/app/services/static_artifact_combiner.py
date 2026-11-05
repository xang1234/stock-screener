from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.atomic_directory_publisher import AtomicDirectoryPublisher
from app.services.static_group_matrix import validate_group_matrix_asset
from app.services.breadth.types import CURRENT_BREADTH_CALCULATION_REVISION
from app.services.market_session_lag import market_session_lag, publication_late_after
from app.services.static_advertised_paths import (
    StaticAdvertisedPathError,
    validate_advertised_paths,
    validate_scoreboard_asset,
)
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


class StaticArtifactDamagedError(RuntimeError):
    """A Market artifact is damaged or breaks the static artifact contract."""


class StaticArtifactIncompatibleError(RuntimeError):
    """A Market artifact was built with an RS formula or breadth revision
    this release does not publish."""


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
        required_markets: Iterable[str] = (),
        optional_markets: Iterable[str] = (),
        clean: bool,
    ) -> StaticArtifactCombineResult:
        required = {
            str(market).strip().upper()
            for market in required_markets
            if str(market).strip()
        }
        optional = {
            str(market).strip().upper()
            for market in optional_markets
            if str(market).strip()
        }
        rejections: list[tuple[str, str]] = []
        current = self._discover(
            Path(artifacts_dir),
            source_label="current",
            rejections=rejections,
        )
        fallback = (
            self._discover(
                Path(fallback_artifacts_dir),
                source_label="fallback",
                rejections=rejections,
            )
            if fallback_artifacts_dir is not None
            else {}
        )
        selected, fallback_reasons = self._select_artifacts(
            current=current,
            fallback=fallback,
        )

        if required:
            missing = sorted(
                market
                for market in required
                if market not in selected and market not in optional
            )
            if missing:
                details = "; ".join(
                    message for market, message in rejections if market in missing
                )
                raise NoPublishedStaticMarketArtifact(
                    "No published compatible static artifact is available for required "
                    f"Markets: {', '.join(missing)}."
                    + (f" Rejected: {details}." if details else ""),
                    markets=tuple(missing),
                )
        elif not selected:
            raise RuntimeError(
                "No market artifacts are available to combine into a static-site bundle"
            )

        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = [
            f"Ignored static market artifact: {message}"
            for _market, message in rejections
        ]
        entries: dict[str, dict[str, Any]] = {}
        for market, artifact in selected.items():
            entries[market] = artifact["entry"]
            entries[market]["publication"] = {
                "source": artifact["source_label"],
                "session_date": artifact["entry"].get("as_of_date"),
            }
            warnings.extend(str(item) for item in artifact["warnings"])
            # With no current artifacts at all (the static-site publisher, which
            # selects stored artifacts) every market is a fallback, so a
            # per-market reuse warning would carry no information.
            if artifact["source_label"] == "fallback" and current:
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
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        selected = dict(current)
        fallback_reasons: dict[str, str] = {}
        for market, fallback_artifact in fallback.items():
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

    def _discover(
        self,
        root: Path,
        *,
        source_label: str,
        rejections: list[tuple[str, str]],
    ) -> dict[str, dict[str, Any]]:
        discovered: dict[str, dict[str, Any]] = {}
        paths = sorted(root.rglob(self._metadata_filename)) if root.exists() else []
        for metadata_path in paths:
            expected_market = expected_market_from_static_market_manifest_path(
                root,
                metadata_path,
            )
            try:
                metadata = read_static_market_manifest(
                    metadata_path,
                    expected_schema_version=self._schema_version,
                    expected_market=expected_market,
                )
            except (OSError, ValueError) as exc:
                # An unreadable manifest is a damaged artifact; contract errors
                # (wrong schema or market) stay fatal, as in the validator.
                rejections.append(
                    (expected_market or "", f"{metadata_path} is unreadable ({exc})")
                )
                continue
            market = str(metadata.get("market") or "").strip().upper()
            if market in discovered:
                raise RuntimeError(f"Duplicate {source_label} artifact for {market}")
            market_dir = metadata_path.parent
            reason = self.rejection_reason(
                market=market,
                source_label=source_label,
                metadata=metadata,
                market_dir=market_dir,
            )
            if reason is not None:
                rejections.append((market, reason))
                continue
            entry, asset_warnings = self._without_broken_breadth_contributors(
                market=market, entry=metadata["entry"], market_dir=market_dir
            )
            entry, scoreboard_warnings = self._without_broken_scoreboard(
                market=market, entry=entry, market_dir=market_dir
            )
            discovered[market] = {
                "entry": entry,
                "warnings": [
                    *metadata.get("warnings", []),
                    *asset_warnings,
                    *scoreboard_warnings,
                ],
                "market_dir": market_dir,
                "source_label": source_label,
            }
        return discovered

    @classmethod
    def rejection_reason(
        cls,
        *,
        market: str,
        source_label: str,
        metadata: dict[str, Any],
        market_dir: Path,
    ) -> str | None:
        """Why a Market artifact cannot be published, or None when it can.

        Checks the RS formula, the breadth revision and the advertised assets
        of an artifact whose manifest is already read. Damage only rejects the
        artifact, so a valid fallback can serve the Market. A current artifact
        built with an incompatible formula or breadth revision raises
        ``StaticArtifactIncompatibleError`` instead: the export just built it,
        so that is a defect to surface. ``metadata`` is left unchanged.
        """
        entry = metadata.get("entry")
        if not isinstance(entry, dict):
            return f"{market} {source_label} metadata has no Market entry"
        try:
            # Shape first: the formula and breadth checks read the payloads the
            # advertised-asset check proves are present and parse.
            cls._validate_advertised_assets(
                market=market,
                source_label=source_label,
                entry=entry,
                market_dir=market_dir,
            )
            cls._validate_formula(
                market=market,
                source_label=source_label,
                entry=entry,
                market_dir=market_dir,
                expected_formula=BALANCED_RS_FORMULA_VERSION,
            )
            cls._validate_breadth_revision(
                market=market,
                source_label=source_label,
                entry=entry,
                market_dir=market_dir,
            )
        except StaticArtifactIncompatibleError as exc:
            if source_label == "current":
                raise
            return str(exc)
        except StaticArtifactDamagedError as exc:
            return str(exc)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return f"{market} {source_label} artifact is unreadable ({exc})"
        return None

    @staticmethod
    def _validate_advertised_assets(
        *, market: str, source_label: str, entry: dict, market_dir: Path
    ) -> None:
        try:
            validate_advertised_paths(
                market=market, entry=entry, market_dir=market_dir
            )
        except StaticAdvertisedPathError as exc:
            raise StaticArtifactDamagedError(f"{market} {source_label} {exc}") from exc
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        for feature, filename in (
            ("groups", "groups.json"),
            ("rrg", "groups_rrg.json"),
        ):
            if features.get(feature) and not (market_dir / filename).is_file():
                raise StaticArtifactDamagedError(
                    f"{market} {source_label} artifact advertises "
                    f"{feature.upper()} but {filename} is absent"
                )
        try:
            validate_group_matrix_asset(market=market, market_dir=market_dir, entry=entry)
        except (ValueError, OSError) as exc:
            raise StaticArtifactDamagedError(f"{market} invalid Matrix asset: {exc}") from exc

    @classmethod
    def _without_broken_breadth_contributors(
        cls, *, market: str, entry: dict, market_dir: Path
    ) -> tuple[dict, list[str]]:
        assets = entry.get("assets")
        contributor_asset = (
            assets.get("breadth_contributors") if isinstance(assets, dict) else None
        )
        if contributor_asset is None:
            return entry, []
        try:
            cls._validate_breadth_contributor_asset(
                market=market,
                market_dir=market_dir,
                descriptor=contributor_asset,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            StaticArtifactDamagedError,
            StaticBreadthContributorAssetError,
        ) as exc:
            kept_assets = {
                name: asset
                for name, asset in assets.items()
                if name != "breadth_contributors"
            }
            return (
                {**entry, "assets": kept_assets},
                [f"{market} breadth contributor asset ignored: {exc}"],
            )
        return entry, []

    @staticmethod
    def _without_broken_scoreboard(
        *, market: str, entry: dict, market_dir: Path
    ) -> tuple[dict, list[str]]:
        assets = entry.get("assets")
        if not isinstance(assets, dict) or "scoreboard" not in assets:
            return entry, []
        try:
            validate_scoreboard_asset(
                market=market, market_dir=market_dir, descriptor=assets["scoreboard"]
            )
        except StaticAdvertisedPathError as exc:
            kept_assets = {
                name: asset for name, asset in assets.items() if name != "scoreboard"
            }
            return (
                {**entry, "assets": kept_assets},
                [f"{market} scoreboard asset ignored: {exc}"],
            )
        return entry, []

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
            raise StaticArtifactDamagedError(str(exc)) from exc

    @staticmethod
    def _read_object(path: Path, label: str) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise StaticArtifactDamagedError(f"{label} is not a JSON object")
        return payload

    @classmethod
    def _validate_formula(
        cls,
        *,
        market: str,
        source_label: str,
        entry: dict,
        market_dir: Path,
        expected_formula: str,
    ) -> None:
        observed = {"market entry": entry.get("rs_formula_version")}
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        scan_manifest_path = market_dir / "scan" / "manifest.json"
        if features.get("scan") and not scan_manifest_path.is_file():
            raise StaticArtifactDamagedError(
                f"{market} {source_label} artifact advertises Scan but "
                "scan/manifest.json is absent"
            )
        if scan_manifest_path.is_file():
            scan_manifest = cls._read_object(
                scan_manifest_path, f"{market} {source_label} Scan manifest"
            )
            observed["Scan manifest"] = scan_manifest.get("rs_formula_version")
            chunk_refs = scan_manifest.get("chunks") or []
            if not isinstance(chunk_refs, list) or not all(
                isinstance(ref, dict) for ref in chunk_refs
            ):
                raise StaticArtifactDamagedError(
                    f"{market} {source_label} Scan manifest must list chunks as objects"
                )
            artifact_root = market_dir.resolve()
            published_market_prefix = Path("markets") / market.lower()
            for chunk_ref in chunk_refs:
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
                    raise StaticArtifactDamagedError(
                        f"{market} {source_label} Scan chunk path escapes its artifact: "
                        f"{advertised_path!r}"
                    ) from exc
                if not advertised_path or not chunk_path.is_file():
                    raise StaticArtifactDamagedError(
                        f"{market} {source_label} Scan chunk is absent: "
                        f"{advertised_path!r}"
                    )
                chunk = cls._read_object(
                    chunk_path, f"{market} {source_label} Scan chunk {chunk_path.name}"
                )
                observed[f"Scan chunk {chunk_path.name}"] = chunk.get(
                    "rs_formula_version"
                )
        groups_path = market_dir / "groups.json"
        if features.get("groups") and not groups_path.is_file():
            raise StaticArtifactDamagedError(
                f"{market} {source_label} artifact advertises Groups but groups.json is absent"
            )
        if groups_path.is_file():
            groups = cls._read_object(
                groups_path, f"{market} {source_label} groups.json"
            )
            if groups.get("available", True):
                observed["Groups"] = groups.get("rs_formula_version")
        rrg_path = market_dir / "groups_rrg.json"
        if features.get("rrg") and not rrg_path.is_file():
            raise StaticArtifactDamagedError(
                f"{market} {source_label} artifact advertises RRG but groups_rrg.json is absent"
            )
        if rrg_path.is_file():
            rrg = cls._read_object(rrg_path, f"{market} {source_label} groups_rrg.json")
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
            raise StaticArtifactIncompatibleError(
                f"{market} {source_label} artifact uses incompatible RS formula: "
                f"{rendered}; expected {expected_formula!r}"
            )

    @staticmethod
    def _validate_breadth_revision(
        *,
        market: str,
        source_label: str,
        entry: dict,
        market_dir: Path,
    ) -> None:
        features = (
            entry.get("features") if isinstance(entry.get("features"), dict) else {}
        )
        if not features.get("breadth"):
            return

        breadth_path = market_dir / "breadth.json"
        if not breadth_path.is_file():
            raise StaticArtifactDamagedError(
                f"{market} {source_label} artifact advertises breadth but "
                "breadth.json is absent"
            )
        breadth = StaticArtifactCombiner._read_object(
            breadth_path, f"{market} {source_label} breadth.json"
        )
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
            raise StaticArtifactIncompatibleError(
                f"{market} {source_label} artifact uses incompatible breadth revision: "
                f"revision={observed_revision!r}, source_revision={source_revision!r}; "
                f"expected {CURRENT_BREADTH_CALCULATION_REVISION} and marker "
                f"{expected_marker!r}"
            )

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
            # Markets with no usable artifact stay listed so the site can say
            # so instead of silently dropping them; they advertise no data path.
            "unavailable_markets": [
                market
                for market in self._supported_markets
                if market not in market_entries
            ],
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
                market_stage = stage / "markets" / market.lower()
                shutil.copytree(artifact["market_dir"], market_stage, dirs_exist_ok=True)
                # The publisher parses every JSON file it ships, so a scoreboard
                # dropped for damage would fail the whole site if it stayed.
                if "scoreboard" not in artifact["entry"].get("assets", {}):
                    (market_stage / "scoreboard.json").unlink(missing_ok=True)
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        AtomicDirectoryPublisher().publish(
            output_dir,
            populate,
            clean=clean,
        )


def annotate_publication_lag(manifest: dict[str, Any], calendar: Any) -> None:
    """Add session lag and state to each served market's publication block.

    ``state`` is ``current`` at lag 0, ``stale`` above it, and ``unknown``
    when the session date or the market calendar cannot decide. Freshness is
    judged against the last completed session, so a run during a session that
    serves today's partial bar counts as current. ``late_after``: see
    ``publication_late_after`` (#503).
    """
    for market, entry in (manifest.get("markets") or {}).items():
        publication = entry.setdefault("publication", {})
        publication["late_after"] = publication_late_after(calendar, market, publication.get("session_date"))
        try:
            session = date.fromisoformat(str(publication.get("session_date"))[:10])
            lag: int | None = max(
                0,
                market_session_lag(
                    calendar,
                    market=market,
                    start_date=session,
                    end_date=calendar.last_completed_trading_day(market),
                ),
            )
        except Exception:  # noqa: BLE001 - unknown freshness must not block publication
            lag = None
        publication["session_lag"] = lag
        publication["state"] = (
            "unknown" if lag is None else "current" if lag == 0 else "stale"
        )

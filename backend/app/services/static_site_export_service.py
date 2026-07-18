"""Static site export service for the daily GitHub Pages build."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any, Mapping
from urllib.parse import quote

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from app.analysis.patterns.rs_line import blue_dot_series, compute_rs_line
from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.markets.catalog import get_market_catalog
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.domain.scanning.default_filters import (
    DEFAULT_SCAN_FILTERS_BY_MARKET,
    DEFAULT_SCAN_FILTERS_FALLBACK,
    resolve_default_scan_filters,
)
from app.infra.serialization import json_safe
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.models.industry import IBDGroupRank
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.breadth_attribution_service import BreadthAttributionService
from app.services.group_detail_payloads import (
    constituent_stock_payloads_from_group_rows,
)
from app.services.group_ranking_history import build_group_detail_payload_from_parts
from app.services.group_ranking_payloads import group_snapshot_metadata
from app.services.key_market_history import build_key_market_entries
from app.services.market_exposure_service import build_exposure_payload
from app.services.preset_screens import (
    PRESET_SCREENS,
    get_preset_chart_symbols,
    resolve_preset_screens_for_defaults,
)
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGUnavailableError,
    StaticGroupsRRGPayloadSource,
)
from app.services.static_groups_payload_builder import (
    StaticGroupsSnapshot,
    build_static_groups_payload,
)
from app.services.snapshot_date_coherence import (
    SnapshotSectionDates,
    build_snapshot_freshness,
)
from app.services.ui_snapshot_service import UISnapshotService
from app.wiring.bootstrap import (
    get_benchmark_cache,
    get_fundamentals_cache,
    get_group_rank_service,
    get_price_cache,
)


logger = logging.getLogger(__name__)

STATIC_SITE_SCHEMA_VERSION = "static-site-v3"
SCAN_BUNDLE_SCHEMA_VERSION = "static-scan-v1"
CHART_BUNDLE_SCHEMA_VERSION = "static-charts-v1"
SCAN_CHUNK_SIZE = 1000
STATIC_CHART_LIMIT = 200
STATIC_CHART_PERIOD = "6mo"
STATIC_CHART_PERIOD_DAYS = 180
STATIC_CHART_LOOKUP_BATCH_SIZE = 250
# Canonical per-market defaults live in the domain layer (shared with the
# Daily Snapshot service); these aliases preserve this module's public names.
STATIC_DEFAULT_SCAN_FILTERS_BY_MARKET = DEFAULT_SCAN_FILTERS_BY_MARKET
STATIC_DEFAULT_SCAN_FILTERS_FALLBACK = DEFAULT_SCAN_FILTERS_FALLBACK


STATIC_CHART_PRESET_TOP_N = 200
STATIC_CHART_TOP_N_GROUPS = 50
STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS = 10
STATIC_BREADTH_ATTRIBUTION_MARKETS = ("US",)
STATIC_DEFAULT_MARKET = "US"
_MARKET_CATALOG = get_market_catalog()
STATIC_SUPPORTED_MARKETS = tuple(_MARKET_CATALOG.supported_market_codes())
STATIC_MARKET_METADATA_FILENAME = "manifest.market.json"
STATIC_MARKET_DISPLAY = {
    market: _MARKET_CATALOG.get(market).label for market in STATIC_SUPPORTED_MARKETS
}
STATIC_GROUP_HISTORY_RUNS = 40
@dataclass(frozen=True)
class StaticSiteExportResult:
    """Summary of one static-site export run."""

    output_dir: Path
    generated_at: str
    as_of_date: str
    warnings: tuple[str, ...]
    manifest: dict[str, Any]


class NoPublishedStaticMarketArtifact(RuntimeError):
    """Raised when static export cannot find a published market artifact source."""

    def __init__(self, message: str, *, markets: tuple[str, ...] = ()) -> None:
        self.markets = tuple(markets)
        super().__init__(message)


class StaticSiteSectionUnavailableError(RuntimeError):
    """Raised when an optional static-site section cannot be exported for the target date."""

    def __init__(self, *, section: str, reason: str) -> None:
        self.section = section
        self.reason = reason
        super().__init__(reason)


class StaticSiteExportService:
    """Generate a static JSON bundle for the read-only frontend."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        rrg_payload_source: StaticGroupsRRGPayloadSource | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._rrg_payload_source = (
            rrg_payload_source
            if rrg_payload_source is not None
            else StaticGroupsRRGDatabasePayloadSource(
                schema_version=STATIC_SITE_SCHEMA_VERSION,
            )
        )
        self._ui_snapshot_service = UISnapshotService(session_factory)
        self._market_rs_repository = MarketRsRunRepository()
        self._price_cache = get_price_cache()
        self._fundamentals_cache = get_fundamentals_cache()
        self._benchmark_cache = get_benchmark_cache()

    def export(
        self,
        output_dir: Path,
        *,
        clean: bool = True,
        markets: tuple[str, ...] | None = None,
        write_manifest: bool = True,
        rs_formula_version_overrides: Mapping[str, str] | None = None,
        feature_run_ids_by_market: Mapping[str, int] | None = None,
    ) -> StaticSiteExportResult:
        output_dir = Path(output_dir)
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._session_factory() as db:
            market_entries: dict[str, dict[str, Any]] = {}
            selected_markets = tuple(
                str(market).strip().upper()
                for market in (markets or STATIC_SUPPORTED_MARKETS)
            )
            formula_overrides = {
                str(market).strip().upper(): str(formula).strip()
                for market, formula in (rs_formula_version_overrides or {}).items()
            }
            run_overrides = {
                str(market).strip().upper(): int(run_id)
                for market, run_id in (feature_run_ids_by_market or {}).items()
            }
            available_markets = [
                market
                for market in selected_markets
                if self._resolve_static_feature_run(
                    db,
                    market=market,
                    feature_run_id=run_overrides.get(market),
                )
                is not None
            ]
            if not available_markets:
                latest_run = self._get_latest_published_run(db)
                if latest_run is None:
                    raise NoPublishedStaticMarketArtifact(
                        "No published feature run is available for static-site export",
                        markets=selected_markets,
                    )
                raise NoPublishedStaticMarketArtifact(
                    "No market-scoped published feature runs are available for static-site export",
                    markets=selected_markets,
                )

            for market in available_markets:
                warning_count_before = len(warnings)
                market_entries[market] = self._export_market_bundle(
                    db=db,
                    output_dir=output_dir,
                    market=market,
                    generated_at=generated_at,
                    warnings=warnings,
                    formula_version_override=formula_overrides.get(market),
                    feature_run_id=run_overrides.get(market),
                )
                self._write_market_metadata(
                    output_dir=output_dir,
                    generated_at=generated_at,
                    market=market,
                    entry=market_entries[market],
                    warnings=warnings[warning_count_before:],
                )

        manifest = self._build_manifest(
            market_entries=market_entries,
            generated_at=generated_at,
            warnings=warnings,
        )
        if write_manifest:
            self._write_json(output_dir / "manifest.json", manifest)

        return StaticSiteExportResult(
            output_dir=output_dir,
            generated_at=generated_at,
            as_of_date=manifest["as_of_date"],
            warnings=tuple(warnings),
            manifest=manifest,
        )

    @classmethod
    def combine_market_artifacts(
        cls,
        artifacts_dir: Path,
        output_dir: Path,
        *,
        fallback_artifacts_dir: Path | None = None,
        clean: bool = True,
        rs_formula_version_overrides: Mapping[str, str] | None = None,
    ) -> StaticSiteExportResult:
        artifacts_dir = Path(artifacts_dir)
        output_dir = Path(output_dir)
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []
        formula_overrides = {
            str(market).strip().upper(): str(formula).strip()
            for market, formula in (rs_formula_version_overrides or {}).items()
        }

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        market_entries, warnings = cls._collect_market_artifacts(
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            warnings=warnings,
            allow_empty=True,
            expected_formula_by_market=formula_overrides,
        )
        if fallback_artifacts_dir is not None:
            fallback_entries, warnings = cls._collect_market_artifacts(
                artifacts_dir=Path(fallback_artifacts_dir),
                output_dir=output_dir,
                warnings=warnings,
                skip_markets=set(market_entries),
                allow_empty=True,
                fallback_source=True,
                expected_formula_by_market=formula_overrides,
            )
            market_entries.update(fallback_entries)
        if not market_entries:
            raise RuntimeError("No market artifacts are available to combine into a static-site bundle")
        missing_markets = [
            market for market in STATIC_SUPPORTED_MARKETS if market not in market_entries
        ]
        warnings.extend(
            f"Static export market {market} was omitted from the combined bundle because no artifact was produced."
            for market in missing_markets
        )
        manifest = cls._build_manifest(
            market_entries=market_entries,
            generated_at=generated_at,
            warnings=warnings,
        )
        cls._write_json(output_dir / "manifest.json", manifest)
        return StaticSiteExportResult(
            output_dir=output_dir,
            generated_at=generated_at,
            as_of_date=manifest["as_of_date"],
            warnings=tuple(warnings),
            manifest=manifest,
        )

    def _export_market_bundle(
        self,
        *,
        db: Session,
        output_dir: Path,
        market: str,
        generated_at: str,
        warnings: list[str],
        formula_version_override: str | None = None,
        feature_run_id: int | None = None,
    ) -> dict[str, Any]:
        latest_run = self._resolve_static_feature_run(
            db,
            market=market,
            feature_run_id=feature_run_id,
        )
        if latest_run is None:
            raise NoPublishedStaticMarketArtifact(
                f"No published feature run is available for static-site export market {market}",
                markets=(market,),
            )

        path_prefix = Path("markets") / market.lower()
        scan_rows, filter_options = self._load_scan_export_source(db, latest_run)
        formula_version = (
            str(formula_version_override).strip()
            if formula_version_override is not None
            else self._market_rs_repository.active_formula(db, market=market)
        )
        scan_manifest, serialized_rows = self._export_scan_bundle(
            db=db,
            output_dir=output_dir,
            generated_at=generated_at,
            run=latest_run,
            rows=scan_rows,
            filter_options=filter_options,
            path_prefix=path_prefix,
            market=market,
        )
        groups_payload = self._build_optional_section_payload(
            section=f"{market} groups",
            warnings=warnings,
            generated_at=generated_at,
            expected_as_of_date=latest_run.as_of_date,
            build=lambda: self._build_groups_payload(
                db=db,
                generated_at=generated_at,
                expected_as_of_date=latest_run.as_of_date,
                market=market,
                latest_run=latest_run,
                formula_version=formula_version,
                serialized_rows=serialized_rows,
            ),
        )
        rrg_payload = self._build_optional_section_payload(
            section=f"{market} rrg",
            warnings=warnings,
            generated_at=generated_at,
            expected_as_of_date=latest_run.as_of_date,
            build=lambda: self._build_groups_rrg_payload(
                db=db,
                generated_at=generated_at,
                expected_as_of_date=latest_run.as_of_date,
                market=market,
                formula_version=formula_version,
            ),
        )
        chart_manifest = self._export_chart_bundle(
            output_dir=output_dir,
            generated_at=generated_at,
            run=latest_run,
            rows=scan_rows,
            serialized_rows=serialized_rows,
            path_prefix=path_prefix,
            groups_payload=groups_payload,
            preset_screens=scan_manifest.get("preset_screens"),
        )
        breadth_payload = self._build_optional_section_payload(
            section=f"{market} breadth",
            warnings=warnings,
            generated_at=generated_at,
            expected_as_of_date=latest_run.as_of_date,
            build=lambda: self._build_breadth_payload(
                generated_at=generated_at,
                expected_as_of_date=latest_run.as_of_date,
                market=market,
                serialized_rows=serialized_rows,
            ),
        )
        home_payload = self._build_home_payload(
            db=db,
            generated_at=generated_at,
            latest_run=latest_run,
            market=market,
            scan_manifest=scan_manifest,
            breadth_payload=breadth_payload,
            groups_payload=groups_payload,
        )

        scan_manifest["charts"] = {
            "path": chart_manifest["path"],
            "limit": chart_manifest["limit"],
            "symbols_total": chart_manifest["symbols_total"],
            "available": chart_manifest["available"],
        }
        self._write_json(output_dir / path_prefix / "scan" / "manifest.json", scan_manifest)

        skipped_chart_symbols = chart_manifest.get("skipped_symbols") or []
        if skipped_chart_symbols:
            preview = ", ".join(skipped_chart_symbols[:5])
            warnings.append(
                f"Static charts skipped {len(skipped_chart_symbols)} {market} symbols without cached "
                f"{STATIC_CHART_PERIOD} price history" + (f": {preview}" if preview else "")
            )

        breadth_path = path_prefix / "breadth.json"
        groups_path = path_prefix / "groups.json"
        groups_rrg_path = path_prefix / "groups_rrg.json"
        home_path = path_prefix / "home.json"
        rrg_available = bool(rrg_payload.get("available", False))
        self._write_json(output_dir / breadth_path, breadth_payload)
        self._write_json(output_dir / groups_path, groups_payload)
        self._write_json(output_dir / home_path, home_payload)

        assets: dict[str, Any] = {
            "charts": {
                "path": chart_manifest["path"],
                "limit": chart_manifest["limit"],
                "symbols_total": chart_manifest["symbols_total"],
            },
        }
        # Only publish the RRG asset/file for markets that actually have it, so
        # the static page hides the RRG toggle (gated on assets.groups_rrg.path)
        # instead of offering an empty view that triggers a wasted fetch.
        if rrg_available:
            self._write_json(output_dir / groups_rrg_path, rrg_payload)
            assets["groups_rrg"] = {"path": groups_rrg_path.as_posix()}

        return {
            "market": market,
            "display_name": STATIC_MARKET_DISPLAY.get(market, market),
            "as_of_date": latest_run.as_of_date.isoformat(),
            "rs_formula_version": formula_version,
            "market_rs_run_id": groups_payload.get("market_rs_run_id"),
            "rs_as_of_date": groups_payload.get(
                "rs_as_of_date", latest_run.as_of_date.isoformat()
            ),
            "rs_universe_size": groups_payload.get("rs_universe_size"),
            "features": {
                "scan": True,
                "breadth": bool(breadth_payload.get("available", True)),
                "groups": bool(groups_payload.get("available", False)),
                "rrg": rrg_available,
                "charts": bool(chart_manifest.get("available", False)),
            },
            "pages": {
                "home": {"path": home_path.as_posix()},
                "scan": {"path": (path_prefix / "scan" / "manifest.json").as_posix()},
                "breadth": {"path": breadth_path.as_posix()},
                "groups": {"path": groups_path.as_posix()},
            },
            "assets": assets,
            "freshness": home_payload.get("freshness", {}),
        }

    @staticmethod
    def _market_metadata_path(market: str) -> Path:
        return Path("markets") / market.lower() / STATIC_MARKET_METADATA_FILENAME

    def _write_market_metadata(
        self,
        *,
        output_dir: Path,
        generated_at: str,
        market: str,
        entry: dict[str, Any],
        warnings: list[str],
    ) -> None:
        payload = {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "market": market,
            "entry": entry,
            "warnings": list(warnings),
        }
        self._write_json(output_dir / self._market_metadata_path(market), payload)

    @staticmethod
    def _build_manifest(
        *,
        market_entries: dict[str, dict[str, Any]],
        generated_at: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        if not market_entries:
            raise RuntimeError("No market artifacts are available to build a static-site manifest")

        ordered_markets = [
            market for market in STATIC_SUPPORTED_MARKETS if market in market_entries
        ]
        ordered_entries = {market: market_entries[market] for market in ordered_markets}
        default_market = (
            STATIC_DEFAULT_MARKET
            if STATIC_DEFAULT_MARKET in ordered_entries
            else next(iter(ordered_entries))
        )
        default_entry = ordered_entries[default_market]
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
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

    @classmethod
    def _collect_market_artifacts(
        cls,
        *,
        artifacts_dir: Path,
        output_dir: Path,
        warnings: list[str],
        skip_markets: set[str] | None = None,
        allow_empty: bool = False,
        fallback_source: bool = False,
        expected_formula_by_market: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        market_entries: dict[str, dict[str, Any]] = {}
        skipped = skip_markets or set()
        metadata_paths = sorted(artifacts_dir.rglob(STATIC_MARKET_METADATA_FILENAME)) if artifacts_dir.exists() else []
        for metadata_path in metadata_paths:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            market = str(payload["market"]).upper()
            schema_version = payload.get("schema_version")
            if schema_version != STATIC_SITE_SCHEMA_VERSION:
                message = (
                    f"{market} artifact uses schema_version {schema_version!r}; "
                    f"expected {STATIC_SITE_SCHEMA_VERSION!r}."
                )
                if fallback_source:
                    warnings.append(
                        f"{market} fallback artifact uses schema_version {schema_version!r}; "
                        f"expected {STATIC_SITE_SCHEMA_VERSION!r}. Skipping."
                    )
                    continue
                raise RuntimeError(f"Invalid market metadata payload at {metadata_path}: {message}")
            if market in skipped:
                continue
            entry = payload.get("entry")
            if not isinstance(entry, dict):
                raise RuntimeError(f"Invalid market metadata payload at {metadata_path}")
            expected_formula = (expected_formula_by_market or {}).get(market)
            actual_formula = entry.get("rs_formula_version")
            if expected_formula is not None and actual_formula != expected_formula:
                message = (
                    f"{market} artifact uses RS formula {actual_formula!r}; "
                    f"expected {expected_formula!r}."
                )
                if fallback_source:
                    warnings.append(f"{message} Skipping fallback artifact.")
                    continue
                raise RuntimeError(
                    f"Invalid market metadata payload at {metadata_path}: {message}"
                )
            source_market_dir = metadata_path.parent
            target_market_dir = output_dir / "markets" / market.lower()
            shutil.copytree(source_market_dir, target_market_dir, dirs_exist_ok=True)
            market_entries[market] = entry
            warnings.extend(str(item) for item in payload.get("warnings", []))
            if fallback_source:
                warnings.append(
                    f"{market} reused from a previous static-site market artifact because the current run produced no artifact."
                )

        if not market_entries and not allow_empty:
            raise RuntimeError("No market artifacts are available to combine into a static-site bundle")
        return market_entries, warnings

    def _build_groups_rrg_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
        formula_version: str,
    ) -> dict[str, Any]:
        """Pre-compute the Relative Rotation Graph payload for the static bundle.

        There is no live API in static mode, so RRG coordinates are baked here
        using the SAME pure math as the live endpoint (``RRGService`` ->
        ``compute_group_rrg``), emitting the same ``{date, market, scope,
        groups[]}`` shape the shared ``RRGChart`` consumes. Both scopes
        (groups + sectors) are stored so the static page's toggle works offline.

        RRG tails want ~30 weekly points (~7 months) of
        ``avg_rs_rating`` history — when the exported DB is shallower, the math
        flags ``is_provisional`` / omits thin groups rather than fabricating.
        If a lightweight export database lacks the RRG source tables entirely,
        this optional section is reported unavailable without aborting export.
        """
        try:
            return self._rrg_payload_source.build(
                db=db,
                generated_at=generated_at,
                expected_as_of_date=expected_as_of_date,
                market=market,
                formula_version=formula_version,
            )
        except StaticGroupsRRGUnavailableError as exc:
            raise StaticSiteSectionUnavailableError(
                section=exc.section,
                reason=exc.reason,
            ) from exc

    def _build_optional_section_payload(
        self,
        *,
        section: str,
        warnings: list[str],
        generated_at: str,
        expected_as_of_date: date,
        build,
    ) -> dict[str, Any]:
        try:
            return build()
        except StaticSiteSectionUnavailableError as exc:
            warnings.append(
                f"Static {section} data unavailable for {expected_as_of_date.isoformat()}: {exc.reason}"
            )
            return self._build_unavailable_payload(
                generated_at=generated_at,
                expected_as_of_date=expected_as_of_date,
                message=exc.reason,
            )

    @staticmethod
    def _build_unavailable_payload(
        *,
        generated_at: str,
        expected_as_of_date: date,
        message: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": False,
            "expected_as_of_date": expected_as_of_date.isoformat(),
            "message": message,
            "payload": {},
        }

    def _get_latest_published_run(self, db: Session, market: str | None = None) -> FeatureRun | None:
        normalized_market = market.upper() if market is not None else None
        pointer_key = (
            f"latest_published_market:{normalized_market}"
            if market is not None
            else "latest_published"
        )
        pointer = (
            db.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == pointer_key)
            .first()
        )
        if pointer is not None:
            run = db.query(FeatureRun).filter(FeatureRun.id == pointer.run_id).first()
            if (
                run is not None
                and run.status == "published"
                and (normalized_market is None or feature_run_market(run) == normalized_market)
            ):
                return run

        query = (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.published_at.desc(), FeatureRun.id.desc())
        )
        if market is None:
            return query.first()
        for run in query.all():
            if feature_run_market(run) == normalized_market:
                return run
        return None

    def _resolve_static_feature_run(
        self,
        db: Session,
        *,
        market: str,
        feature_run_id: int | None,
    ) -> FeatureRun | None:
        normalized_market = str(market or "").strip().upper()
        if feature_run_id is None:
            return self._get_latest_published_run(db, market=normalized_market)
        run = db.get(FeatureRun, int(feature_run_id))
        if run is None:
            raise NoPublishedStaticMarketArtifact(
                f"Feature run {feature_run_id} does not exist for static market {normalized_market}",
                markets=(normalized_market,),
            )
        if run.status != "published":
            raise NoPublishedStaticMarketArtifact(
                f"Feature run {feature_run_id} is not published for static market {normalized_market}",
                markets=(normalized_market,),
            )
        if feature_run_market(run) != normalized_market:
            raise NoPublishedStaticMarketArtifact(
                f"Feature run {feature_run_id} is not scoped to static market {normalized_market}",
                markets=(normalized_market,),
            )
        return run

    def _load_scan_export_rows(
        self,
        db: Session,
        run: FeatureRun,
        *,
        include_sparklines: bool,
    ) -> list[Any]:
        repo = SqlFeatureStoreRepository(db)
        return repo.query_all_as_scan_results(
            run.id,
            FilterSpec(),
            SortSpec(field="composite_score", order=SortOrder.DESC),
            include_sparklines=include_sparklines,
        )

    def _load_scan_export_source(self, db: Session, run: FeatureRun) -> tuple[list[Any], Any]:
        repo = SqlFeatureStoreRepository(db)
        rows = self._load_scan_export_rows(
            db,
            run,
            include_sparklines=True,
        )
        filter_options = repo.get_filter_options_for_run(run.id)
        return rows, filter_options

    def _export_scan_bundle(
        self,
        *,
        db: Session,
        output_dir: Path,
        generated_at: str,
        run: FeatureRun,
        rows: list[Any] | None = None,
        filter_options: Any | None = None,
        path_prefix: Path | None = None,
        market: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if rows is None or filter_options is None:
            repo = SqlFeatureStoreRepository(db)
            rows = repo.query_all_as_scan_results(
                run.id,
                FilterSpec(),
                SortSpec(field="composite_score", order=SortOrder.DESC),
                include_sparklines=True,
            )
            filter_options = repo.get_filter_options_for_run(run.id)

        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        scan_dir = output_dir / normalized_prefix / "scan"
        chunk_dir = scan_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        serialized_rows = [self._serialize_scan_row(row) for row in rows]
        self._annotate_percentile_ranks(serialized_rows)
        serialized_rows = self._sort_static_scan_rows(serialized_rows)
        resolved_default_filters = self.resolve_static_default_filters(market)
        resolved_preset_screens = resolve_preset_screens_for_defaults(
            PRESET_SCREENS,
            resolved_default_filters,
        )
        default_filtered_rows = self._apply_static_default_filters(
            serialized_rows, default_filters=resolved_default_filters
        )
        chunk_refs: list[dict[str, Any]] = []
        for index in range(0, len(serialized_rows), SCAN_CHUNK_SIZE):
            chunk_rows = serialized_rows[index:index + SCAN_CHUNK_SIZE]
            chunk_num = (index // SCAN_CHUNK_SIZE) + 1
            rel_path = normalized_prefix / "scan" / "chunks" / f"chunk-{chunk_num:04d}.json"
            payload = {
                "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "as_of_date": run.as_of_date.isoformat(),
                "run_id": run.id,
                "chunk_index": chunk_num,
                "rows": chunk_rows,
            }
            self._write_json(output_dir / rel_path, payload)
            chunk_refs.append(
                {
                    "path": rel_path.as_posix(),
                    "count": len(chunk_rows),
                }
            )

        manifest = {
            "schema_version": SCAN_BUNDLE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": run.as_of_date.isoformat(),
            "run_id": run.id,
            "sort": {"field": "composite_score", "order": "desc"},
            "default_page_size": 50,
            "chunk_size": SCAN_CHUNK_SIZE,
            "rows_total": len(serialized_rows),
            "default_filters": dict(resolved_default_filters),
            "default_filtered_rows_total": len(default_filtered_rows),
            "filter_options": FilterOptionsResponse(
                ibd_industries=list(filter_options.ibd_industries),
                gics_sectors=list(filter_options.gics_sectors),
                ratings=list(filter_options.ratings),
            ).model_dump(mode="json"),
            "preset_screens": resolved_preset_screens,
            "chunks": chunk_refs,
            "initial_rows": default_filtered_rows[:50],
            "preview_rows": default_filtered_rows[:10],
        }
        self._write_json(scan_dir / "manifest.json", manifest)
        return manifest, serialized_rows

    def _export_chart_bundle(
        self,
        *,
        output_dir: Path,
        generated_at: str,
        run: FeatureRun,
        rows: list[Any],
        serialized_rows: list[dict[str, Any]] | None = None,
        path_prefix: Path | None = None,
        groups_payload: dict[str, Any] | None = None,
        preset_screens: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        chart_dir = output_dir / normalized_prefix / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        # Market benchmark (fetched once) drives the per-symbol RS line + blue dots.
        market = feature_run_market(run) or STATIC_DEFAULT_MARKET
        benchmark_symbol, benchmark_df = self._get_market_benchmark_history(market, period="2y")

        entries: list[dict[str, Any]] = []
        skipped_symbols: list[str] = []
        row_by_symbol: dict[str, Any] = {}
        ordered_rows = list(rows)

        def _emit_chart(symbol, *, rank, stock_data, price_df, fundamentals_value) -> None:
            """Serialize + write one chart payload, recording it in ``entries`` (or skip)."""
            bars = self._serialize_chart_bars(price_df)
            if not bars:
                skipped_symbols.append(symbol)
                return
            rs_line, blue_dots = self._serialize_rs_line(price_df, benchmark_df)
            rel_path = self._chart_payload_path(symbol, path_prefix=normalized_prefix)
            self._write_json(
                output_dir / rel_path,
                {
                    "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "as_of_date": run.as_of_date.isoformat(),
                    "symbol": symbol,
                    "rank": rank,
                    "period": STATIC_CHART_PERIOD,
                    "bars": bars,
                    "rs_line": rs_line,
                    "blue_dots": blue_dots,
                    "benchmark_symbol": benchmark_symbol,
                    "stock_data": stock_data,
                    "fundamentals": fundamentals_value,
                },
            )
            entries.append({"symbol": symbol, "rank": rank, "path": rel_path.as_posix()})

        def _expand_extra_charts(candidate_symbols, *, log_label) -> None:
            """Emit charts for symbols not already exported (preset/group expansion)."""
            extra = sorted(candidate_symbols - {e["symbol"] for e in entries} - set(skipped_symbols))
            if not extra:
                return
            before = len(entries)
            ser_by_symbol = {r["symbol"]: r for r in (serialized_rows or []) if r.get("symbol")}
            for row in ordered_rows:
                sym = getattr(row, "symbol", None)
                if sym and sym not in row_by_symbol:
                    row_by_symbol[sym] = row
            for offset in range(0, len(extra), STATIC_CHART_LOOKUP_BATCH_SIZE):
                batch = extra[offset:offset + STATIC_CHART_LOOKUP_BATCH_SIZE]
                price_data = self._price_cache.get_many_cached_only(batch, period="2y")
                fundamentals = self._fundamentals_cache.get_many_cached_only(batch)
                for symbol in batch:
                    domain_row = row_by_symbol.get(symbol)
                    stock_data = self._serialize_scan_row(domain_row) if domain_row else ser_by_symbol.get(symbol)
                    _emit_chart(
                        symbol,
                        rank=None,
                        stock_data=stock_data,
                        price_df=price_data.get(symbol),
                        fundamentals_value=fundamentals.get(symbol),
                    )
            logger.info(
                "%s added %d charts (%d extra symbols attempted)",
                log_label,
                len(entries) - before,
                len(extra),
            )

        if serialized_rows is not None:
            raw_rows_by_symbol = {
                getattr(row, "symbol", None): row
                for row in rows
                if getattr(row, "symbol", None)
            }
            ordered_symbols = [row["symbol"] for row in serialized_rows if row.get("symbol")]
            ordered_rows = [
                raw_rows_by_symbol[symbol]
                for symbol in ordered_symbols
                if symbol in raw_rows_by_symbol
            ]
            seen_symbols = {getattr(row, "symbol", None) for row in ordered_rows}
            ordered_rows.extend(
                row
                for row in rows
                if getattr(row, "symbol", None) not in seen_symbols
            )

        # --- Pass 1: export charts for top-N by composite score (default) ---
        for start in range(0, len(ordered_rows), STATIC_CHART_LOOKUP_BATCH_SIZE):
            if len(entries) >= STATIC_CHART_LIMIT:
                break

            batch_rows = list(ordered_rows[start:start + STATIC_CHART_LOOKUP_BATCH_SIZE])
            symbols = [row.symbol for row in batch_rows if getattr(row, "symbol", None)]
            price_data = self._price_cache.get_many_cached_only(symbols, period="2y")
            fundamentals = self._fundamentals_cache.get_many_cached_only(symbols)

            for rank, row in enumerate(batch_rows, start=start + 1):
                if len(entries) >= STATIC_CHART_LIMIT:
                    break

                symbol = getattr(row, "symbol", None)
                if not symbol:
                    continue

                row_by_symbol[symbol] = row
                _emit_chart(
                    symbol,
                    rank=rank,
                    stock_data=self._serialize_scan_row(row),
                    price_df=price_data.get(symbol),
                    fundamentals_value=fundamentals.get(symbol),
                )

        # --- Pass 2: expand preset screen top-N charts ---
        if serialized_rows is not None:
            _expand_extra_charts(
                get_preset_chart_symbols(
                    serialized_rows,
                    PRESET_SCREENS if preset_screens is None else preset_screens,
                    STATIC_CHART_PRESET_TOP_N,
                ),
                log_label="Preset screen expansion",
            )

        # --- Pass 3: expand coverage to all constituents of top-N groups ---
        group_symbols = self._collect_top_group_constituent_symbols(
            groups_payload=groups_payload,
            top_n=STATIC_CHART_TOP_N_GROUPS,
        )
        if group_symbols:
            _expand_extra_charts(
                group_symbols,
                log_label=f"Top-{STATIC_CHART_TOP_N_GROUPS} groups expansion",
            )

        index_rel_path = normalized_prefix / "charts" / "index.json"
        index_payload = {
            "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": run.as_of_date.isoformat(),
            "limit": STATIC_CHART_LIMIT,
            "symbols_total": len(entries),
            "skipped_symbols": skipped_symbols,
            "symbols": entries,
        }
        self._write_json(output_dir / index_rel_path, index_payload)
        return {
            "path": index_rel_path.as_posix(),
            "limit": STATIC_CHART_LIMIT,
            "symbols_total": len(entries),
            "available": bool(entries),
            "skipped_symbols": skipped_symbols,
        }

    @staticmethod
    def _collect_top_group_constituent_symbols(
        *,
        groups_payload: dict[str, Any] | None,
        top_n: int,
    ) -> set[str]:
        """Return constituent symbols across the top-N IBD industry groups.

        Pulls from `groups_payload['payload']['group_details']`, which maps
        group name → details (including `current_rank` and `stocks`).
        """
        if not groups_payload or not groups_payload.get("available"):
            return set()
        payload = groups_payload.get("payload") or {}
        group_details = payload.get("group_details") or {}
        symbols: set[str] = set()
        for detail in group_details.values():
            if not isinstance(detail, dict):
                continue
            rank = detail.get("current_rank")
            if rank is None or rank > top_n:
                continue
            for stock in detail.get("stocks") or []:
                if isinstance(stock, dict):
                    symbol = stock.get("symbol")
                    if symbol:
                        symbols.add(symbol)
        return symbols

    def _build_breadth_payload(
        self,
        *,
        generated_at: str,
        expected_as_of_date: date,
        market: str = STATIC_DEFAULT_MARKET,
        serialized_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if serialized_rows is None:
            snapshot = self._ui_snapshot_service.publish_breadth_bootstrap(market=market).to_dict()
            payload = snapshot.get("payload", {})
            current_date = ((payload.get("current") or {}).get("date"))
            if current_date != expected_as_of_date.isoformat():
                raise StaticSiteSectionUnavailableError(
                    section="breadth",
                    reason=(
                        "No breadth snapshot is available for static-site export date "
                        f"{expected_as_of_date.isoformat()} (latest snapshot date: {current_date or 'none'})."
                    ),
                )
            return {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "available": True,
                "published_at": _coerce_datetime(snapshot.get("published_at")),
                "source_revision": snapshot.get("source_revision"),
                "payload": payload,
            }

        symbols = [row["symbol"] for row in serialized_rows if row.get("symbol")]
        if not symbols:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No scan rows are available for market {market} on {expected_as_of_date.isoformat()}.",
            )

        benchmark_symbol, benchmark = self._get_market_benchmark_history(market, period="1y")
        if benchmark.empty:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No cached benchmark price history is available for market {market}.",
            )

        canonical_dates = [
            ts.date()
            for ts in pd.to_datetime(benchmark.index)
            if ts.date() <= expected_as_of_date
        ]
        if expected_as_of_date not in canonical_dates:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=(
                    f"No benchmark trading session is available for market {market} "
                    f"on {expected_as_of_date.isoformat()}."
                ),
            )

        canonical_dates = canonical_dates[-max(STATIC_BREADTH_HISTORY_LOOKBACK_DAYS + 15, 120):]
        price_data = self._get_cached_price_histories(symbols, period="1y")
        metrics_by_date = self._compute_breadth_metrics_by_date(canonical_dates, price_data)
        current = metrics_by_date.get(expected_as_of_date)
        if current is None:
            raise StaticSiteSectionUnavailableError(
                section="breadth",
                reason=f"No breadth snapshot could be derived for market {market} on {expected_as_of_date.isoformat()}.",
            )

        ordered_dates = sorted(metrics_by_date.keys())
        ordered_history = [
            {**metrics_by_date[item_date], "market": market}
            for item_date in ordered_dates
        ]
        chart_data = ordered_history[-31:]
        current = {**current, "market": market}
        benchmark_overlay = self._serialize_history_bars(
            benchmark,
            period_days=31,
            end_date=expected_as_of_date,
        )
        group_attribution = self._build_group_attribution(
            market=market,
            serialized_rows=serialized_rows,
            price_data=price_data,
            ordered_dates=ordered_dates,
        )
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": True,
            "published_at": _coerce_datetime(datetime.utcnow()),
            "source_revision": f"feature-run:{market}:{expected_as_of_date.isoformat()}",
            "market": market,
            "payload": {
                "current": current,
                "summary": {
                    "market": market,
                    "latest_date": expected_as_of_date.isoformat(),
                    "total_records": len(ordered_history),
                    "date_range_start": ordered_dates[0].isoformat() if ordered_dates else None,
                    "date_range_end": ordered_dates[-1].isoformat() if ordered_dates else None,
                },
                "history_90d": list(reversed(ordered_history[-STATIC_BREADTH_HISTORY_LOOKBACK_DAYS:])),
                "chart_range": "1M",
                "chart_data": list(reversed(chart_data)),
                "benchmark_symbol": benchmark_symbol,
                "benchmark_overlay": benchmark_overlay,
                "spy_overlay": benchmark_overlay,
                "group_attribution": group_attribution,
            },
        }

    def _build_group_attribution(
        self,
        *,
        market: str,
        serialized_rows: list[dict[str, Any]],
        price_data: dict[str, pd.DataFrame | None],
        ordered_dates: list[date],
    ) -> dict[str, Any]:
        """Attribute ±4% movers to IBD industry groups for the most recent sessions.

        Only enabled for markets in ``STATIC_BREADTH_ATTRIBUTION_MARKETS`` — non-US
        taxonomies aren't wired in for the first cut. Returns an
        ``{available: False, reason}`` payload when skipped so the static client
        can hide the feature cleanly.
        """
        if market not in STATIC_BREADTH_ATTRIBUTION_MARKETS:
            return {
                "available": False,
                "reason": f"Group attribution is not yet supported for market {market}.",
            }
        if not ordered_dates:
            return {
                "available": False,
                "reason": "No trading dates were available to attribute.",
            }

        attribution_dates = ordered_dates[-STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS:]
        symbols_meta = [
            {
                "symbol": row.get("symbol"),
                "company_name": row.get("company_name"),
                "ibd_industry_group": row.get("ibd_industry_group"),
            }
            for row in serialized_rows
            if row.get("symbol")
        ]
        service = BreadthAttributionService()
        history = service.compute(
            symbols_meta=symbols_meta,
            price_data=price_data,
            target_dates=attribution_dates,
        )
        has_any_mover = any(
            (day.get("stocks_up_4pct", 0) + day.get("stocks_down_4pct", 0)) > 0
            for day in history
        )
        if not history or not has_any_mover:
            return {
                "available": False,
                "reason": "No 4%+ movers were attributable for the lookback window.",
            }

        latest = history[-1]
        return {
            "available": True,
            "market": market,
            "threshold_pct": 4.0,
            "lookback_days": STATIC_BREADTH_ATTRIBUTION_LOOKBACK_DAYS,
            "latest_date": latest["date"] if latest else None,
            "history": list(reversed(history)),
        }

    def _build_groups_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        expected_as_of_date: date,
        market: str,
        latest_run: FeatureRun,
        formula_version: str,
        serialized_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_market = str(market or "").strip().upper()
        group_service = get_group_rank_service()
        rankings = group_service.get_current_rankings(
            db,
            limit=197,
            calculation_date=expected_as_of_date,
            market=normalized_market,
            formula_version=formula_version,
        )
        if not rankings:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    "No exact stored Group snapshot is available for "
                    f"{normalized_market} on {expected_as_of_date.isoformat()} "
                    f"with formula {formula_version}."
                ),
            )
        try:
            rs_metadata = group_snapshot_metadata(
                db,
                market=normalized_market,
                rankings=rankings,
            )
        except RuntimeError as exc:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=str(exc),
            ) from exc

        market_rs_run_id = rankings[0].get("market_rs_run_id")
        self._validate_feature_run_group_source(
            latest_run=latest_run,
            market=normalized_market,
            expected_as_of_date=expected_as_of_date,
            formula_version=formula_version,
            market_rs_run_id=market_rs_run_id,
            rs_universe_size=rs_metadata["rs_universe_size"],
            serialized_rows=serialized_rows,
        )
        historical_rankings = self._load_stored_group_history(
            db,
            group_service=group_service,
            market=normalized_market,
            formula_version=formula_version,
            through_date=expected_as_of_date,
            current_rankings=rankings,
        )
        group_details = self._build_stored_group_details(
            rankings=rankings,
            serialized_rows=serialized_rows,
            historical_rankings=historical_rankings,
        )
        movers = self._build_group_movers(rankings)

        return build_static_groups_payload(
            StaticGroupsSnapshot(
                date=expected_as_of_date.isoformat(),
                rankings=rankings,
                movers=movers,
                group_details=group_details,
                market=normalized_market,
                rs_formula_version=rs_metadata["rs_formula_version"],
                market_rs_run_id=market_rs_run_id,
                rs_as_of_date=rs_metadata["rs_as_of_date"],
                rs_universe_size=rs_metadata["rs_universe_size"],
            ),
            generated_at=generated_at,
            schema_version=STATIC_SITE_SCHEMA_VERSION,
        )

    @staticmethod
    def _validate_feature_run_group_source(
        *,
        latest_run: FeatureRun,
        market: str,
        expected_as_of_date: date,
        formula_version: str,
        market_rs_run_id: int | None,
        rs_universe_size: int | None,
        serialized_rows: list[dict[str, Any]],
    ) -> None:
        config = latest_run.config_json or {}
        if (
            latest_run.status != "published"
            or latest_run.as_of_date != expected_as_of_date
            or feature_run_market(latest_run) != market
        ):
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} does not match published "
                    f"{market} snapshot date {expected_as_of_date.isoformat()}."
                ),
            )

        configured_formula = config.get("rs_formula_version")
        configured_run_id = config.get("market_rs_run_id")
        configured_as_of = config.get("rs_as_of_date")
        configured_universe = config.get("rs_universe_size")
        if configured_formula is not None and configured_formula != formula_version:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} formula {configured_formula} does not "
                    f"match Group formula {formula_version}."
                ),
            )
        if formula_version == BALANCED_RS_FORMULA_VERSION:
            required = {
                "rs_formula_version": configured_formula,
                "market_rs_run_id": configured_run_id,
                "rs_as_of_date": configured_as_of,
                "rs_universe_size": configured_universe,
            }
            missing = sorted(key for key, value in required.items() if value is None)
            if missing:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=(
                        f"Feature run {latest_run.id} is missing canonical RS metadata: "
                        f"{', '.join(missing)}."
                    ),
                )
        if configured_run_id is not None and int(configured_run_id) != market_rs_run_id:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} Market RS run {configured_run_id} does not "
                    f"match Group Market RS run {market_rs_run_id}."
                ),
            )
        if configured_as_of is not None and str(configured_as_of) != expected_as_of_date.isoformat():
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} RS date {configured_as_of} does not match "
                    f"Group date {expected_as_of_date.isoformat()}."
                ),
            )
        if configured_universe is not None and int(configured_universe) != rs_universe_size:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    f"Feature run {latest_run.id} RS universe {configured_universe} does not "
                    f"match Group RS universe {rs_universe_size}."
                ),
            )
        for row in serialized_rows:
            row_formula = row.get("rs_formula_version")
            row_run_id = row.get("market_rs_run_id")
            if row_formula is not None and row_formula != formula_version:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=f"Stock row {row.get('symbol')} uses a different RS formula.",
                )
            if row_run_id is not None and int(row_run_id) != market_rs_run_id:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=f"Stock row {row.get('symbol')} uses a different Market RS run.",
                )

    @staticmethod
    def _load_stored_group_history(
        db: Session,
        *,
        group_service: Any,
        market: str,
        formula_version: str,
        through_date: date,
        current_rankings: list[dict[str, Any]],
    ) -> list[tuple[date, list[dict[str, Any]]]]:
        dates = [
            row[0]
            for row in (
                db.query(IBDGroupRank.date)
                .filter(
                    IBDGroupRank.market == market,
                    IBDGroupRank.rs_formula_version == formula_version,
                    IBDGroupRank.date <= through_date,
                )
                .distinct()
                .order_by(IBDGroupRank.date.desc())
                .limit(STATIC_GROUP_HISTORY_RUNS)
                .all()
            )
        ]
        history: list[tuple[date, list[dict[str, Any]]]] = []
        for ranking_date in dates:
            rows = (
                current_rankings
                if ranking_date == through_date
                else group_service.get_current_rankings(
                    db,
                    limit=197,
                    calculation_date=ranking_date,
                    market=market,
                    formula_version=formula_version,
                )
            )
            history.append((ranking_date, rows))
        return history

    @staticmethod
    def _build_stored_group_details(
        *,
        rankings: list[dict[str, Any]],
        serialized_rows: list[dict[str, Any]],
        historical_rankings: list[tuple[date, list[dict[str, Any]]]],
    ) -> dict[str, Any]:
        current_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in serialized_rows:
            group = row.get("ibd_industry_group")
            if group:
                current_rows_by_group[str(group)].append(row)
        historical_maps = [
            (
                ranking_date,
                {str(row["industry_group"]): row for row in rows},
            )
            for ranking_date, rows in historical_rankings
        ]
        details: dict[str, Any] = {}
        for ranking in rankings:
            group = str(ranking["industry_group"])
            history = []
            for _ranking_date, rows_by_group in historical_maps:
                row = rows_by_group.get(group)
                if row is None:
                    continue
                history.append(
                    {
                        "date": row["date"],
                        "rank": row["rank"],
                        "avg_rs_rating": row["avg_rs_rating"],
                        "avg_rs_rating_1m": row.get("avg_rs_rating_1m"),
                        "avg_rs_rating_3m": row.get("avg_rs_rating_3m"),
                        "num_stocks": row["num_stocks"],
                    }
                )
            details[group] = build_group_detail_payload_from_parts(
                group,
                ranking=ranking,
                history=history,
                stocks=constituent_stock_payloads_from_group_rows(
                    current_rows_by_group.get(group, [])
                ),
            )
        return details

    def _build_home_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        latest_run: FeatureRun,
        market: str,
        scan_manifest: dict[str, Any],
        breadth_payload: dict[str, Any],
        groups_payload: dict[str, Any],
    ) -> dict[str, Any]:
        key_markets = build_key_market_entries(
            db,
            market,
            as_of_date=latest_run.as_of_date,
        )
        top_groups = (
            ((groups_payload.get("payload") or {}).get("rankings") or {}).get("rankings") or []
        )[:10]

        breadth_current = ((breadth_payload.get("payload") or {}).get("current")) or {}
        exposure_payload = build_exposure_payload(db, market, as_of_date=latest_run.as_of_date)
        exposure_date = (
            exposure_payload.get("date")
            if isinstance(exposure_payload, dict)
            else None
        )
        snapshot_date = latest_run.as_of_date.isoformat()
        breadth_date = breadth_current.get("date")
        groups_date = (((groups_payload.get("payload") or {}).get("rankings") or {}).get("date"))
        freshness = build_snapshot_freshness(
            base_freshness={
                "scan_run_id": latest_run.id,
                "scan_as_of_date": snapshot_date,
                "scan_published_at": _coerce_datetime(latest_run.published_at),
            },
            anchor=latest_run.as_of_date,
            market_timezone=get_market_catalog().get(market).display_timezone,
            section_dates=SnapshotSectionDates.from_raw(
                breadth=breadth_date,
                groups=groups_date,
                exposure=exposure_date,
            ),
            key_markets=key_markets,
        )

        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": snapshot_date,
            "market": market,
            "market_display_name": STATIC_MARKET_DISPLAY.get(market, market),
            "freshness": freshness,
            "key_markets": key_markets,
            "market_health_exposure": exposure_payload,
            "scan_summary": {
                "run_id": latest_run.id,
                "rows_total": scan_manifest.get("rows_total", 0),
                "default_filtered_rows_total": scan_manifest.get("default_filtered_rows_total", 0),
                "top_results": scan_manifest.get("preview_rows", []),
            },
            "top_groups": top_groups,
        }

    @staticmethod
    def _build_group_movers(rankings: list[dict[str, Any]]) -> dict[str, Any]:
        gainers = sorted(
            [row for row in rankings if (row.get("rank_change_1w") or 0) > 0],
            key=lambda row: (-(row.get("rank_change_1w") or 0), row["rank"]),
        )[:10]
        losers = sorted(
            [row for row in rankings if (row.get("rank_change_1w") or 0) < 0],
            key=lambda row: ((row.get("rank_change_1w") or 0), row["rank"]),
        )[:10]
        return {"period": "1w", "gainers": gainers, "losers": losers}

    def _get_cached_price_histories(
        self,
        symbols: list[str],
        *,
        period: str,
    ) -> dict[str, pd.DataFrame | None]:
        results: dict[str, pd.DataFrame | None] = {}
        for start in range(0, len(symbols), STATIC_CHART_LOOKUP_BATCH_SIZE):
            batch = symbols[start:start + STATIC_CHART_LOOKUP_BATCH_SIZE]
            results.update(self._price_cache.get_many_cached_only(batch, period=period))
        return results

    def _get_market_benchmark_history(self, market: str, *, period: str) -> tuple[str, pd.DataFrame]:
        for candidate in self._benchmark_cache.get_benchmark_candidates(market):
            history = self._get_symbol_price_history(candidate, period=period)
            if history is not None and not history.empty:
                return candidate, history
        return self._benchmark_cache.get_benchmark_symbol(market), pd.DataFrame()

    def _get_symbol_price_history(self, symbol: str, *, period: str) -> pd.DataFrame | None:
        data = self._price_cache.get_cached_only(symbol.upper(), period=period)
        if data is None or data.empty:
            return None
        return data

    def _compute_breadth_metrics_by_date(
        self,
        canonical_dates: list[date],
        price_data: dict[str, pd.DataFrame | None],
    ) -> dict[date, dict[str, Any]]:
        if not canonical_dates:
            return {}

        date_index = pd.Index(canonical_dates)
        empty = lambda: [0] * len(canonical_dates)
        aggregates = {
            "stocks_up_4pct": empty(),
            "stocks_down_4pct": empty(),
            "stocks_up_25pct_quarter": empty(),
            "stocks_down_25pct_quarter": empty(),
            "stocks_up_25pct_month": empty(),
            "stocks_down_25pct_month": empty(),
            "stocks_up_50pct_month": empty(),
            "stocks_down_50pct_month": empty(),
            "stocks_up_13pct_34days": empty(),
            "stocks_down_13pct_34days": empty(),
            "total_stocks_scanned": empty(),
        }

        for history in price_data.values():
            if history is None or history.empty or "Close" not in history.columns:
                continue
            close_series = pd.Series(
                history["Close"].to_numpy(),
                index=[ts.date() for ts in pd.to_datetime(history.index)],
            )
            close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()
            pct_1d = ((close_series / close_series.shift(1)) - 1.0) * 100.0
            pct_21d = ((close_series / close_series.shift(21)) - 1.0) * 100.0
            pct_34d = ((close_series / close_series.shift(34)) - 1.0) * 100.0
            pct_63d = ((close_series / close_series.shift(63)) - 1.0) * 100.0

            close_series = close_series.reindex(date_index)
            pct_1d = pct_1d.reindex(date_index)
            pct_21d = pct_21d.reindex(date_index)
            pct_34d = pct_34d.reindex(date_index)
            pct_63d = pct_63d.reindex(date_index)
            valid = close_series.notna().to_numpy()

            for index, is_valid in enumerate(valid):
                if is_valid:
                    aggregates["total_stocks_scanned"][index] += 1

            for key, series in (
                ("stocks_up_4pct", pct_1d >= 4.0),
                ("stocks_down_4pct", pct_1d <= -4.0),
                ("stocks_up_25pct_month", pct_21d >= 25.0),
                ("stocks_down_25pct_month", pct_21d <= -25.0),
                ("stocks_up_50pct_month", pct_21d >= 50.0),
                ("stocks_down_50pct_month", pct_21d <= -50.0),
                ("stocks_up_13pct_34days", pct_34d >= 13.0),
                ("stocks_down_13pct_34days", pct_34d <= -13.0),
                ("stocks_up_25pct_quarter", pct_63d >= 25.0),
                ("stocks_down_25pct_quarter", pct_63d <= -25.0),
            ):
                flags = series.fillna(False).to_numpy()
                for index, flag in enumerate(flags):
                    if flag and valid[index]:
                        aggregates[key][index] += 1

        results: dict[date, dict[str, Any]] = {}
        for index, item_date in enumerate(canonical_dates):
            ratio_5day = None
            ratio_10day = None
            if index >= 5:
                up_5 = sum(aggregates["stocks_up_4pct"][max(index - 5, 0):index])
                down_5 = sum(aggregates["stocks_down_4pct"][max(index - 5, 0):index])
                ratio_5day = round(up_5 / down_5, 2) if down_5 > 0 else None
            if index >= 10:
                up_10 = sum(aggregates["stocks_up_4pct"][index - 10:index])
                down_10 = sum(aggregates["stocks_down_4pct"][index - 10:index])
                ratio_10day = round(up_10 / down_10, 2) if down_10 > 0 else None

            results[item_date] = {
                "date": item_date.isoformat(),
                "stocks_up_4pct": int(aggregates["stocks_up_4pct"][index]),
                "stocks_down_4pct": int(aggregates["stocks_down_4pct"][index]),
                "ratio_5day": ratio_5day,
                "ratio_10day": ratio_10day,
                "stocks_up_25pct_quarter": int(aggregates["stocks_up_25pct_quarter"][index]),
                "stocks_down_25pct_quarter": int(aggregates["stocks_down_25pct_quarter"][index]),
                "stocks_up_25pct_month": int(aggregates["stocks_up_25pct_month"][index]),
                "stocks_down_25pct_month": int(aggregates["stocks_down_25pct_month"][index]),
                "stocks_up_50pct_month": int(aggregates["stocks_up_50pct_month"][index]),
                "stocks_down_50pct_month": int(aggregates["stocks_down_50pct_month"][index]),
                "stocks_up_13pct_34days": int(aggregates["stocks_up_13pct_34days"][index]),
                "stocks_down_13pct_34days": int(aggregates["stocks_down_13pct_34days"][index]),
                "total_stocks_scanned": int(aggregates["total_stocks_scanned"][index]),
            }
        return results

    @staticmethod
    def _serialize_close_history(data: pd.DataFrame | None, *, days: int) -> list[dict[str, Any]]:
        if data is None or data.empty or "Close" not in data.columns:
            return []
        frame = data.tail(days).reset_index()
        date_col = frame.columns[0]
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        return [
            {
                "date": row["Date"],
                "close": round(float(row["Close"]), 2),
            }
            for _, row in frame.iterrows()
            if row["Close"] is not None and not math.isnan(float(row["Close"]))
        ]

    @staticmethod
    def _serialize_history_bars(
        data: pd.DataFrame | None,
        *,
        period_days: int,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        if data is None or data.empty:
            return []
        end_timestamp = pd.Timestamp(end_date or datetime.utcnow())
        cutoff_date = end_timestamp - timedelta(days=period_days)
        if data.index.tz is not None:
            cutoff_date = cutoff_date.tz_localize(data.index.tz)
            end_timestamp = end_timestamp.tz_localize(data.index.tz)
        filtered = data[(data.index >= cutoff_date) & (data.index <= end_timestamp)]
        if filtered.empty:
            return []
        frame = filtered.reset_index()
        date_col = frame.columns[0]
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
        return [
            {
                "date": row["Date"],
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            for _, row in frame.iterrows()
            if all(
                value is not None and not math.isnan(float(value))
                for value in (row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])
            )
        ]

    def _serialize_scan_row(self, row) -> dict[str, Any]:
        item = ScanResultItem.from_domain(row, include_setup_payload=False).model_dump(mode="json")
        extended = row.extended_fields or {}
        item.update(
            {
                "perf_week": extended.get("perf_week"),
                "perf_month": extended.get("perf_month"),
                "perf_3m": extended.get("perf_3m"),
                "perf_6m": extended.get("perf_6m"),
                "gap_percent": extended.get("gap_percent"),
                "volume_surge": extended.get("volume_surge"),
                "ema_10_distance": extended.get("ema_10_distance"),
                "ema_20_distance": extended.get("ema_20_distance"),
                "ema_50_distance": extended.get("ema_50_distance"),
                "week_52_high_distance": extended.get("week_52_high_distance"),
                "week_52_low_distance": extended.get("week_52_low_distance"),
            }
        )
        return item

    @staticmethod
    def _annotate_percentile_ranks(rows: list[dict[str, Any]]) -> None:
        """Add pct_day/pct_week/pct_month (0-100) to each row in-place."""
        if not rows:
            return
        for src_field, dst_field in (
            ("price_change_1d", "pct_day"),
            ("perf_week", "pct_week"),
            ("perf_month", "pct_month"),
        ):
            ranked = sorted(
                (
                    (i, row[src_field])
                    for i, row in enumerate(rows)
                    if row.get(src_field) is not None
                ),
                key=lambda pair: pair[1],
            )
            total = len(ranked)
            for row in rows:
                row[dst_field] = None
            pos = 0
            while pos < total:
                end = pos
                value = ranked[pos][1]
                while end + 1 < total and ranked[end + 1][1] == value:
                    end += 1
                percentile = round(((end + 1) / total) * 100, 2)
                for idx in range(pos, end + 1):
                    row_idx, _ = ranked[idx]
                    rows[row_idx][dst_field] = percentile
                pos = end + 1

    @staticmethod
    def resolve_static_default_filters(
        market: str | None,
    ) -> dict[str, int | None]:
        """Return the per-market default scan filters, or the no-op fallback."""

        return resolve_default_scan_filters(market)

    @staticmethod
    def _apply_static_default_filters(
        rows: list[dict[str, Any]],
        *,
        default_filters: dict[str, int | None] | None = None,
    ) -> list[dict[str, Any]]:
        filters = default_filters or STATIC_DEFAULT_SCAN_FILTERS_FALLBACK
        min_volume = filters.get("minVolume")
        if min_volume is None:
            return list(rows)
        return [
            row
            for row in rows
            if row.get("volume") is not None and row["volume"] >= min_volume
        ]

    @classmethod
    def _sort_static_scan_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def _score_key(row: dict[str, Any]) -> float:
            score = row.get("composite_score")
            if score is None:
                return float("-inf")
            return float(score)

        return sorted(
            rows,
            key=lambda row: (
                -_score_key(row),
                row.get("symbol") or "",
            ),
        )

    @staticmethod
    def _static_chart_cutoff(index) -> datetime:
        """The display-window cutoff (``STATIC_CHART_PERIOD_DAYS``), converted to ``index``'s tz.

        Computed in UTC and tz-converted (not ``replace(tzinfo=...)``, which would
        reinterpret the UTC wall time as local and shift the boundary day).
        """
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=STATIC_CHART_PERIOD_DAYS)
        index_tz = getattr(index, "tz", None)
        if index_tz is not None:
            return cutoff.tz_convert(index_tz).to_pydatetime()
        return cutoff.tz_localize(None).to_pydatetime()

    def _serialize_rs_line(
        self, stock_data, benchmark_df
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """RS line series + blue-dot dates for a symbol, trimmed to the bar window.

        New-high detection runs over the full cached history (the 252-day lookback
        needs more than the displayed window), then output is trimmed to the same
        ``STATIC_CHART_PERIOD_DAYS`` window as the candles so the overlay aligns.
        """
        if (
            stock_data is None
            or getattr(stock_data, "empty", True)
            or benchmark_df is None
            or benchmark_df.empty
            or "Close" not in benchmark_df.columns
        ):
            return [], []

        rs_line_full = compute_rs_line(stock_data["Close"], benchmark_df["Close"], normalize=True)
        blue_full = blue_dot_series(stock_data["Close"], benchmark_df["Close"])

        cutoff = self._static_chart_cutoff(rs_line_full.index)
        rs_window = rs_line_full[rs_line_full.index >= cutoff].dropna()
        blue_window = blue_full[(blue_full.index >= cutoff) & blue_full]

        rs_line = [
            {"time": ts.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for ts, v in rs_window.items()
        ]
        blue_dots = [ts.strftime("%Y-%m-%d") for ts in blue_window.index]
        return rs_line, blue_dots

    def _serialize_chart_bars(self, data) -> list[dict[str, Any]]:
        if data is None or getattr(data, "empty", True):
            return []

        filtered = data[data.index >= self._static_chart_cutoff(data.index)]
        if filtered.empty:
            return []

        frame = filtered.reset_index()
        date_col = frame.columns[0]
        frame = frame.rename(columns={date_col: "Date"})
        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")

        bars: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            bars.append(
                {
                    "date": row["Date"],
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
            )
        return bars

    @staticmethod
    def _chart_payload_path(symbol: str, *, path_prefix: Path | None = None) -> Path:
        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        return normalized_prefix / "charts" / f"{quote(symbol, safe='')}.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                json_safe(payload),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


def _coerce_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)

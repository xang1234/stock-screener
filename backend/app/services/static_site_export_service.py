"""Static site export service for the daily GitHub Pages build."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Mapping

from sqlalchemy.orm import Session, sessionmaker

from app.domain.common.query import SortOrder, SortSpec
from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.markets.catalog import get_market_catalog
from app.domain.relative_strength import GroupSnapshotIdentity
from app.domain.scanning.default_filters import (
    DEFAULT_SCAN_FILTERS_BY_MARKET,
    DEFAULT_SCAN_FILTERS_FALLBACK,
    resolve_default_scan_filters,
)
from app.domain.scanning.filter_expression_model import FilterExpression
from app.infra.serialization import json_safe
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.market_rs_repo import MarketRsRunRepository
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.key_market_history import build_key_market_entries
from app.services.feature_run_rs_identity import resolve_feature_run_rs_identity
from app.services.market_exposure_service import build_exposure_payload
from app.services.preset_screens import (
    PRESET_SCREENS,
    resolve_preset_screens_for_defaults,
)
from app.services.static_groups_rrg_export import (
    StaticGroupsRRGDatabasePayloadSource,
    StaticGroupsRRGUnavailableError,
    StaticGroupsRRGPayloadSource,
)
from app.services.snapshot_date_coherence import (
    SnapshotSectionDates,
    build_snapshot_freshness,
)
from app.services.ui_snapshot_service import UISnapshotService
from app.services.group_rank_snapshot_reader import GroupRankSnapshotReader
from app.services.static_group_section_builder import StaticGroupSectionBuilder
from app.services.static_site_errors import (
    NoPublishedStaticMarketArtifact,
    StaticSiteSectionUnavailableError,
)
from app.services.static_artifact_combiner import (
    StaticArtifactCombiner,
    StaticArtifactFormulaError,
)
from app.services.static_chart_bundle_exporter import (
    StaticChartBundleConfig,
    StaticChartBundleExporter,
)
from app.services.static_breadth_section_builder import StaticBreadthSectionBuilder
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


class StaticSiteExportService:
    """Generate a static JSON bundle for the read-only frontend."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        rrg_payload_source: StaticGroupsRRGPayloadSource | None = None,
        group_section_builder: StaticGroupSectionBuilder | None = None,
        price_cache=None,
        fundamentals_cache=None,
        benchmark_cache=None,
        chart_config: StaticChartBundleConfig | None = None,
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
        self._price_cache = price_cache if price_cache is not None else get_price_cache()
        self._fundamentals_cache = (
            fundamentals_cache
            if fundamentals_cache is not None
            else get_fundamentals_cache()
        )
        self._benchmark_cache = (
            benchmark_cache if benchmark_cache is not None else get_benchmark_cache()
        )
        self._group_section_builder = group_section_builder or StaticGroupSectionBuilder(
            snapshot_reader=GroupRankSnapshotReader(),
            rank_history_reader=get_group_rank_service(),
        )
        self._chart_exporter = StaticChartBundleExporter(
            price_cache=self._price_cache,
            fundamentals_cache=self._fundamentals_cache,
            benchmark_cache=self._benchmark_cache,
            json_writer=self._write_json,
            scan_row_serializer=self._serialize_scan_row,
            config=chart_config,
        )
        self._breadth_builder = StaticBreadthSectionBuilder(
            ui_snapshot_service=self._ui_snapshot_service,
            price_cache=self._price_cache,
            benchmark_cache=self._benchmark_cache,
        )

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
        combined = StaticArtifactCombiner(
            schema_version=STATIC_SITE_SCHEMA_VERSION,
            supported_markets=STATIC_SUPPORTED_MARKETS,
            default_market=STATIC_DEFAULT_MARKET,
            metadata_filename=STATIC_MARKET_METADATA_FILENAME,
        ).combine(
            artifacts_dir=Path(artifacts_dir),
            fallback_artifacts_dir=(
                Path(fallback_artifacts_dir)
                if fallback_artifacts_dir is not None
                else None
            ),
            output_dir=Path(output_dir),
            required_formula_by_market=rs_formula_version_overrides or {},
            clean=clean,
        )
        return StaticSiteExportResult(
            output_dir=combined.output_dir,
            generated_at=combined.generated_at,
            as_of_date=combined.as_of_date,
            warnings=combined.warnings,
            manifest=combined.manifest,
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

        formula_version = (
            str(formula_version_override).strip()
            if formula_version_override is not None
            else self._market_rs_repository.active_formula(db, market=market)
        )
        feature_formula_version = resolve_feature_run_rs_identity(
            latest_run,
            ranking_date=latest_run.as_of_date,
        ).identity.formula_version
        if feature_formula_version != formula_version:
            raise StaticArtifactFormulaError(
                f"Requested RS formula {formula_version} does not match Feature run "
                f"{latest_run.id} formula {feature_formula_version} for {market}"
            )

        path_prefix = Path("markets") / market.lower()
        scan_rows, filter_options = self._load_scan_export_source(db, latest_run)
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
        return self._group_section_builder.build(
            db=db,
            generated_at=generated_at,
            identity=GroupSnapshotIdentity(
                market, expected_as_of_date, formula_version
            ),
            latest_run=latest_run,
            serialized_rows=serialized_rows,
        )

    def _export_chart_bundle(self, **kwargs) -> dict[str, Any]:
        return self._chart_exporter.export(**kwargs)

    def _build_breadth_payload(self, **kwargs) -> dict[str, Any]:
        return self._breadth_builder.build(**kwargs)

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
            FilterExpression(),
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
            rows, filter_options = self._load_scan_export_source(db, run)

        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        scan_dir = output_dir / normalized_prefix / "scan"
        chunk_dir = scan_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        serialized_rows = [self._serialize_scan_row(row) for row in rows]
        publication = resolve_feature_run_rs_identity(
            run,
            ranking_date=run.as_of_date,
        ).publication
        rs_metadata = {
            "rs_formula_version": publication.snapshot.formula_version,
            "market_rs_run_id": publication.market_rs_run_id,
            "rs_as_of_date": publication.snapshot.as_of_date.isoformat(),
            "rs_universe_size": publication.universe_size,
        }
        for serialized_row in serialized_rows:
            serialized_row.update(rs_metadata)
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
                **rs_metadata,
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
            **rs_metadata,
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
        market_entry = get_market_catalog().get(market)
        breadth_date = breadth_current.get("date")
        groups_date = (((groups_payload.get("payload") or {}).get("rankings") or {}).get("date"))
        freshness = build_snapshot_freshness(
            base_freshness={
                "scan_run_id": latest_run.id,
                "scan_as_of_date": snapshot_date,
                "scan_published_at": _coerce_datetime(latest_run.published_at),
            },
            anchor=latest_run.as_of_date,
            market_timezone=market_entry.display_timezone,
            section_dates=SnapshotSectionDates.from_raw(
                breadth=breadth_date,
                groups=groups_date,
                exposure=exposure_date,
            ),
            key_markets=key_markets,
            groups_applicable=market_entry.capabilities.group_rankings,
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

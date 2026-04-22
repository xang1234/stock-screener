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
from typing import Any
from urllib.parse import quote

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.models.stock import StockPrice
from app.schemas.groups import (
    ConstituentStock,
    GroupDetailResponse,
    GroupRankResponse,
    GroupRankingsResponse,
    HistoricalDataPoint,
    MoversResponse,
)
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.preset_screens import PRESET_SCREENS, get_preset_chart_symbols
from app.services.ui_snapshot_service import UISnapshotService
from app.wiring.bootstrap import (
    get_benchmark_cache,
    get_fundamentals_cache,
    get_group_rank_service,
    get_price_cache,
)


logger = logging.getLogger(__name__)

STATIC_SITE_SCHEMA_VERSION = "static-site-v2"
SCAN_BUNDLE_SCHEMA_VERSION = "static-scan-v1"
CHART_BUNDLE_SCHEMA_VERSION = "static-charts-v1"
SCAN_CHUNK_SIZE = 1000
STATIC_CHART_LIMIT = 200
STATIC_CHART_PERIOD = "6mo"
STATIC_CHART_PERIOD_DAYS = 180
STATIC_CHART_LOOKUP_BATCH_SIZE = 250
STATIC_DEFAULT_SCAN_FILTERS = {"minVolume": 100_000_000}
STATIC_CHART_PRESET_TOP_N = 200
STATIC_GROUP_DETAIL_HISTORY_DAYS = 100
STATIC_BREADTH_HISTORY_LOOKBACK_DAYS = 90
STATIC_DEFAULT_MARKET = "US"
STATIC_SUPPORTED_MARKETS = ("US", "HK", "IN", "JP", "TW")
STATIC_MARKET_METADATA_FILENAME = "manifest.market.json"
STATIC_MARKET_DISPLAY = {
    "US": "United States",
    "HK": "Hong Kong",
    "IN": "India",
    "JP": "Japan",
    "TW": "Taiwan",
}
STATIC_GROUP_HISTORY_RUNS = 40
STATIC_GROUP_CHANGE_OFFSETS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}
_DEFAULT_KEY_MARKETS = {
    "US": (
        {"symbol": "SPY", "display_name": "S&P 500 ETF", "currency": "USD"},
        {"symbol": "QQQ", "display_name": "Nasdaq 100 ETF", "currency": "USD"},
        {"symbol": "IWM", "display_name": "Russell 2000 ETF", "currency": "USD"},
        {"symbol": "GLD", "display_name": "Gold ETF", "currency": "USD"},
        {"symbol": "TLT", "display_name": "20+ Year Treasury ETF", "currency": "USD"},
    ),
    "HK": (
        {"symbol": "^HSI", "display_name": "Hang Seng Index", "currency": "HKD"},
        {"symbol": "2800.HK", "display_name": "Tracker Fund Hong Kong", "currency": "HKD"},
        {"symbol": "0700.HK", "display_name": "Tencent", "currency": "HKD"},
        {"symbol": "3690.HK", "display_name": "Meituan", "currency": "HKD"},
        {"symbol": "0941.HK", "display_name": "China Mobile", "currency": "HKD"},
    ),
    "IN": (
        {"symbol": "^NSEI", "display_name": "Nifty 50", "currency": "INR"},
        {"symbol": "NIFTYBEES.NS", "display_name": "Nippon India ETF Nifty 50 BeES", "currency": "INR"},
        {"symbol": "RELIANCE.NS", "display_name": "Reliance Industries", "currency": "INR"},
        {"symbol": "TCS.NS", "display_name": "Tata Consultancy Services", "currency": "INR"},
        {"symbol": "HDFCBANK.NS", "display_name": "HDFC Bank", "currency": "INR"},
    ),
    "JP": (
        {"symbol": "^N225", "display_name": "Nikkei 225", "currency": "JPY"},
        {"symbol": "1306.T", "display_name": "TOPIX ETF", "currency": "JPY"},
        {"symbol": "7203.T", "display_name": "Toyota", "currency": "JPY"},
        {"symbol": "6758.T", "display_name": "Sony Group", "currency": "JPY"},
        {"symbol": "9984.T", "display_name": "SoftBank Group", "currency": "JPY"},
    ),
    "TW": (
        {"symbol": "^TWII", "display_name": "TAIEX", "currency": "TWD"},
        {"symbol": "0050.TW", "display_name": "TW50 ETF", "currency": "TWD"},
        {"symbol": "2330.TW", "display_name": "TSMC", "currency": "TWD"},
        {"symbol": "2317.TW", "display_name": "Hon Hai", "currency": "TWD"},
        {"symbol": "2454.TW", "display_name": "MediaTek", "currency": "TWD"},
    ),
}


@dataclass(frozen=True)
class StaticSiteExportResult:
    """Summary of one static-site export run."""

    output_dir: Path
    generated_at: str
    as_of_date: str
    warnings: tuple[str, ...]
    manifest: dict[str, Any]


class StaticSiteSectionUnavailableError(RuntimeError):
    """Raised when an optional static-site section cannot be exported for the target date."""

    def __init__(self, *, section: str, reason: str) -> None:
        self.section = section
        self.reason = reason
        super().__init__(reason)


class StaticSiteExportService:
    """Generate a static JSON bundle for the read-only frontend."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._ui_snapshot_service = UISnapshotService(session_factory)
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
    ) -> StaticSiteExportResult:
        output_dir = Path(output_dir)
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._session_factory() as db:
            market_entries: dict[str, dict[str, Any]] = {}
            selected_markets = tuple(markets or STATIC_SUPPORTED_MARKETS)
            available_markets = [
                market
                for market in selected_markets
                if self._get_latest_published_run(db, market=market) is not None
            ]
            if not available_markets:
                latest_run = self._get_latest_published_run(db)
                if latest_run is None:
                    raise RuntimeError("No published feature run is available for static-site export")
                raise RuntimeError(
                    "No market-scoped published feature runs are available for static-site export"
                )

            for market in available_markets:
                warning_count_before = len(warnings)
                market_entries[market] = self._export_market_bundle(
                    db=db,
                    output_dir=output_dir,
                    market=market,
                    generated_at=generated_at,
                    warnings=warnings,
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
        clean: bool = True,
    ) -> StaticSiteExportResult:
        artifacts_dir = Path(artifacts_dir)
        output_dir = Path(output_dir)
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        market_entries, warnings = cls._collect_market_artifacts(
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            warnings=warnings,
        )
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
    ) -> dict[str, Any]:
        latest_run = self._get_latest_published_run(db, market=market)
        if latest_run is None:
            raise RuntimeError(f"No published feature run is available for static-site export market {market}")

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
        )
        chart_manifest = self._export_chart_bundle(
            output_dir=output_dir,
            generated_at=generated_at,
            run=latest_run,
            rows=scan_rows,
            serialized_rows=serialized_rows,
            path_prefix=path_prefix,
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
                current_rows=scan_rows,
                serialized_rows=serialized_rows,
            ),
        )
        home_payload = self._build_home_payload(
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
        home_path = path_prefix / "home.json"
        self._write_json(output_dir / breadth_path, breadth_payload)
        self._write_json(output_dir / groups_path, groups_payload)
        self._write_json(output_dir / home_path, home_payload)

        return {
            "market": market,
            "display_name": STATIC_MARKET_DISPLAY.get(market, market),
            "as_of_date": latest_run.as_of_date.isoformat(),
            "features": {
                "scan": True,
                "breadth": bool(breadth_payload.get("available", True)),
                "groups": bool(groups_payload.get("available", False)),
                "charts": bool(chart_manifest.get("available", False)),
            },
            "pages": {
                "home": {"path": home_path.as_posix()},
                "scan": {"path": (path_prefix / "scan" / "manifest.json").as_posix()},
                "breadth": {"path": breadth_path.as_posix()},
                "groups": {"path": groups_path.as_posix()},
            },
            "assets": {
                "charts": {
                    "path": chart_manifest["path"],
                    "limit": chart_manifest["limit"],
                    "symbols_total": chart_manifest["symbols_total"],
                },
            },
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
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        market_entries: dict[str, dict[str, Any]] = {}
        metadata_paths = sorted(artifacts_dir.rglob(STATIC_MARKET_METADATA_FILENAME))
        for metadata_path in metadata_paths:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            market = str(payload["market"]).upper()
            entry = payload.get("entry")
            if not isinstance(entry, dict):
                raise RuntimeError(f"Invalid market metadata payload at {metadata_path}")
            source_market_dir = metadata_path.parent
            target_market_dir = output_dir / "markets" / market.lower()
            shutil.copytree(source_market_dir, target_market_dir, dirs_exist_ok=True)
            market_entries[market] = entry
            warnings.extend(str(item) for item in payload.get("warnings", []))

        if not market_entries:
            raise RuntimeError("No market artifacts are available to combine into a static-site bundle")
        return market_entries, warnings

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
                and (normalized_market is None or self._run_market(run) == normalized_market)
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
            if self._run_market(run) == normalized_market:
                return run
        return None

    @staticmethod
    def _run_market(run: FeatureRun) -> str | None:
        config = run.config_json or {}
        if not isinstance(config, dict):
            return None
        universe = config.get("universe")
        if isinstance(universe, dict):
            market = universe.get("market")
            if market:
                return str(market).upper()
        return None

    def _load_scan_export_source(self, db: Session, run: FeatureRun) -> tuple[list[Any], Any]:
        repo = SqlFeatureStoreRepository(db)
        rows = repo.query_all_as_scan_results(
            run.id,
            FilterSpec(),
            SortSpec(field="composite_score", order=SortOrder.DESC),
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

        chunk_refs: list[dict[str, Any]] = []
        serialized_rows = [self._serialize_scan_row(row) for row in rows]
        self._annotate_percentile_ranks(serialized_rows)
        default_filtered_rows = self._apply_static_default_filters(serialized_rows)
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
            "default_filters": dict(STATIC_DEFAULT_SCAN_FILTERS),
            "default_filtered_rows_total": len(default_filtered_rows),
            "filter_options": FilterOptionsResponse(
                ibd_industries=list(filter_options.ibd_industries),
                gics_sectors=list(filter_options.gics_sectors),
                ratings=list(filter_options.ratings),
            ).model_dump(mode="json"),
            "preset_screens": PRESET_SCREENS,
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
    ) -> dict[str, Any]:
        normalized_prefix = Path() if path_prefix is None else Path(path_prefix)
        chart_dir = output_dir / normalized_prefix / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        skipped_symbols: list[str] = []
        row_by_symbol: dict[str, Any] = {}

        # --- Pass 1: export charts for top-N by composite score (default) ---
        for start in range(0, len(rows), STATIC_CHART_LOOKUP_BATCH_SIZE):
            if len(entries) >= STATIC_CHART_LIMIT:
                break

            batch_rows = list(rows[start:start + STATIC_CHART_LOOKUP_BATCH_SIZE])
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
                bars = self._serialize_chart_bars(price_data.get(symbol))
                if not bars:
                    skipped_symbols.append(symbol)
                    continue

                rel_path = self._chart_payload_path(symbol, path_prefix=normalized_prefix)
                payload = {
                    "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
                    "generated_at": generated_at,
                    "as_of_date": run.as_of_date.isoformat(),
                    "symbol": symbol,
                    "rank": rank,
                    "period": STATIC_CHART_PERIOD,
                    "bars": bars,
                    "stock_data": self._serialize_scan_row(row),
                    "fundamentals": fundamentals.get(symbol),
                }
                self._write_json(output_dir / rel_path, payload)
                entries.append(
                    {
                        "symbol": symbol,
                        "rank": rank,
                        "path": rel_path.as_posix(),
                    }
                )

        # --- Pass 2: expand preset screen top-N charts ---
        if serialized_rows is not None:
            preset_symbols = get_preset_chart_symbols(
                serialized_rows, PRESET_SCREENS, STATIC_CHART_PRESET_TOP_N,
            )
            exported_symbols = {e["symbol"] for e in entries}
            extra_symbols = sorted(preset_symbols - exported_symbols - set(skipped_symbols))
            entries_before_pass_2 = len(entries)

            if extra_symbols:
                # Build a lookup from serialized rows for extra symbols
                ser_by_symbol = {r["symbol"]: r for r in serialized_rows if r.get("symbol")}
                # Also need domain rows for _serialize_scan_row
                for row in rows:
                    sym = getattr(row, "symbol", None)
                    if sym and sym not in row_by_symbol:
                        row_by_symbol[sym] = row

                for batch_start in range(0, len(extra_symbols), STATIC_CHART_LOOKUP_BATCH_SIZE):
                    batch = extra_symbols[batch_start:batch_start + STATIC_CHART_LOOKUP_BATCH_SIZE]
                    price_data = self._price_cache.get_many_cached_only(batch, period="2y")
                    fundamentals = self._fundamentals_cache.get_many_cached_only(batch)

                    for symbol in batch:
                        bars = self._serialize_chart_bars(price_data.get(symbol))
                        if not bars:
                            skipped_symbols.append(symbol)
                            continue

                        rel_path = self._chart_payload_path(symbol, path_prefix=normalized_prefix)
                        domain_row = row_by_symbol.get(symbol)
                        stock_data = self._serialize_scan_row(domain_row) if domain_row else ser_by_symbol.get(symbol)
                        payload = {
                            "schema_version": CHART_BUNDLE_SCHEMA_VERSION,
                            "generated_at": generated_at,
                            "as_of_date": run.as_of_date.isoformat(),
                            "symbol": symbol,
                            "rank": None,
                            "period": STATIC_CHART_PERIOD,
                            "bars": bars,
                            "stock_data": stock_data,
                            "fundamentals": fundamentals.get(symbol),
                        }
                        self._write_json(output_dir / rel_path, payload)
                        entries.append(
                            {
                                "symbol": symbol,
                                "rank": None,
                                "path": rel_path.as_posix(),
                            }
                        )

                logger.info(
                    "Preset screen expansion added %d charts (%d extra symbols attempted)",
                    len(entries) - entries_before_pass_2,
                    len(extra_symbols),
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

    def _build_breadth_payload(
        self,
        *,
        generated_at: str,
        expected_as_of_date: date,
        market: str = STATIC_DEFAULT_MARKET,
        serialized_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if serialized_rows is None:
            snapshot = self._ui_snapshot_service.publish_breadth_bootstrap().to_dict()
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

        benchmark = self._get_market_benchmark_history(market, period="1y")
        if benchmark is None or benchmark.empty:
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
        ordered_history = [metrics_by_date[item_date] for item_date in ordered_dates]
        chart_data = ordered_history[-31:]
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
                    "latest_date": expected_as_of_date.isoformat(),
                    "total_records": len(ordered_history),
                    "date_range_start": ordered_dates[0].isoformat() if ordered_dates else None,
                    "date_range_end": ordered_dates[-1].isoformat() if ordered_dates else None,
                },
                "history_90d": list(reversed(ordered_history[-STATIC_BREADTH_HISTORY_LOOKBACK_DAYS:])),
                "chart_range": "1M",
                "chart_data": list(reversed(chart_data)),
                "spy_overlay": self._serialize_history_bars(
                    benchmark,
                    period_days=31,
                    end_date=expected_as_of_date,
                ),
            },
        }

    def _build_groups_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        expected_as_of_date: date,
        market: str | None = None,
        latest_run: FeatureRun | None = None,
        current_rows: list[Any] | None = None,
        serialized_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if market is None or latest_run is None or serialized_rows is None:
            service = get_group_rank_service()
            rankings = service.get_current_rankings(
                db,
                limit=197,
                calculation_date=expected_as_of_date,
            )
            if not rankings:
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=(
                        "No group rankings are available for static-site export date "
                        f"{expected_as_of_date.isoformat()}."
                    ),
                )
            movers = service.get_rank_movers(
                db,
                period="1w",
                limit=10,
                calculation_date=expected_as_of_date,
            )
            ranking_date = rankings[0]["date"]
            if ranking_date != expected_as_of_date.isoformat():
                raise StaticSiteSectionUnavailableError(
                    section="groups",
                    reason=(
                        "Group rankings are stale for static-site export date "
                        f"{expected_as_of_date.isoformat()} (latest ranking date: {ranking_date})."
                    ),
                )
            group_details: dict[str, Any] = {}
            for row in rankings:
                group_name = row["industry_group"]
                try:
                    group_details[group_name] = service.get_group_history(
                        db,
                        group_name,
                        days=STATIC_GROUP_DETAIL_HISTORY_DAYS,
                    )
                except Exception:
                    logger.warning("Failed to export detail for group %s", group_name, exc_info=True)
                    db.rollback()

            return {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "available": True,
                "payload": {
                    "rankings": GroupRankingsResponse(
                        date=ranking_date,
                        total_groups=len(rankings),
                        rankings=[GroupRankResponse(**row) for row in rankings],
                    ).model_dump(mode="json"),
                    "movers_period": "1w",
                    "movers": MoversResponse(
                        period=movers["period"],
                        gainers=[GroupRankResponse(**row) for row in movers.get("gainers", [])],
                        losers=[GroupRankResponse(**row) for row in movers.get("losers", [])],
                    ).model_dump(mode="json"),
                    "group_details": group_details,
                },
            }

        rankings = self._compute_group_rankings_from_serialized_rows(
            serialized_rows,
            ranking_date=expected_as_of_date,
        )
        if not rankings:
            raise StaticSiteSectionUnavailableError(
                section="groups",
                reason=(
                    "No group rankings are available for static-site export date "
                    f"{expected_as_of_date.isoformat()}."
                ),
            )
        market_runs = self._get_market_run_series(db, market, latest_run)
        history_runs = self._select_group_history_runs(market_runs)
        historical_rankings = {
            run.id: self._compute_group_rankings_from_rows(
                self._load_scan_export_source(db, run)[0],
                ranking_date=run.as_of_date,
            )
            for run in history_runs
        }
        self._apply_group_rank_changes(rankings, market_runs, historical_rankings)
        group_details = self._build_group_details(
            rankings=rankings,
            serialized_rows=serialized_rows,
            market_runs=market_runs,
            historical_rankings=historical_rankings,
        )
        movers = self._build_group_movers(rankings)

        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": True,
            "market": market,
            "payload": {
                "rankings": GroupRankingsResponse(
                    date=expected_as_of_date.isoformat(),
                    total_groups=len(rankings),
                    rankings=[GroupRankResponse(**row) for row in rankings],
                ).model_dump(mode="json"),
                "movers_period": "1w",
                "movers": MoversResponse(
                    period=movers["period"],
                    gainers=[GroupRankResponse(**row) for row in movers.get("gainers", [])],
                    losers=[GroupRankResponse(**row) for row in movers.get("losers", [])],
                ).model_dump(mode="json"),
                "group_details": group_details,
            },
        }

    def _build_home_payload(
        self,
        *,
        generated_at: str,
        latest_run: FeatureRun,
        market: str,
        scan_manifest: dict[str, Any],
        breadth_payload: dict[str, Any],
        groups_payload: dict[str, Any],
    ) -> dict[str, Any]:
        key_markets = self._build_key_markets(market)
        top_groups = (
            ((groups_payload.get("payload") or {}).get("rankings") or {}).get("rankings") or []
        )[:10]

        breadth_current = ((breadth_payload.get("payload") or {}).get("current")) or {}

        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": latest_run.as_of_date.isoformat(),
            "market": market,
            "market_display_name": STATIC_MARKET_DISPLAY.get(market, market),
            "freshness": {
                "scan_run_id": latest_run.id,
                "scan_as_of_date": latest_run.as_of_date.isoformat(),
                "scan_published_at": _coerce_datetime(latest_run.published_at),
                "breadth_latest_date": breadth_current.get("date"),
                "groups_latest_date": (((groups_payload.get("payload") or {}).get("rankings") or {}).get("date")),
            },
            "key_markets": key_markets,
            "scan_summary": {
                "run_id": latest_run.id,
                "rows_total": scan_manifest.get("rows_total", 0),
                "default_filtered_rows_total": scan_manifest.get("default_filtered_rows_total", 0),
                "top_results": scan_manifest.get("preview_rows", []),
            },
            "top_groups": top_groups,
        }

    def _build_key_markets(self, market: str | Session = STATIC_DEFAULT_MARKET) -> list[dict[str, Any]]:
        if isinstance(market, Session):
            db = market
            entries: list[dict[str, Any]] = []
            for item in _DEFAULT_KEY_MARKETS[STATIC_DEFAULT_MARKET]:
                rows = (
                    db.query(StockPrice)
                    .filter(StockPrice.symbol == item["symbol"])
                    .order_by(StockPrice.date.desc())
                    .limit(30)
                    .all()
                )
                ordered = list(reversed(rows))
                latest = ordered[-1] if ordered else None
                previous = ordered[-2] if len(ordered) > 1 else None
                change_1d = None
                if (
                    latest is not None
                    and previous is not None
                    and latest.close is not None
                    and previous.close not in (None, 0)
                ):
                    change_1d = round(((latest.close - previous.close) / previous.close) * 100, 2)
                entries.append(
                    {
                        "symbol": item["symbol"],
                        "display_name": item["display_name"],
                        "latest_close": latest.close if latest is not None else None,
                        "latest_date": latest.date.isoformat() if latest is not None else None,
                        "change_1d": change_1d,
                        "history": [
                            {"date": row.date.isoformat(), "close": row.close}
                            for row in ordered
                        ],
                    }
                )
            return entries

        entries: list[dict[str, Any]] = []
        for item in _DEFAULT_KEY_MARKETS.get(market, ()):
            history = self._get_symbol_price_history(item["symbol"], period="6mo")
            ordered = self._serialize_close_history(history, days=30)
            latest = ordered[-1] if ordered else None
            previous = ordered[-2] if len(ordered) > 1 else None
            change_1d = None
            if latest is not None and previous is not None and previous.get("close") not in (None, 0):
                change_1d = round(((latest["close"] - previous["close"]) / previous["close"]) * 100, 2)
            entries.append(
                {
                    "symbol": item["symbol"],
                    "display_name": item["display_name"],
                    "currency": item.get("currency"),
                    "latest_close": latest["close"] if latest is not None else None,
                    "latest_date": latest["date"] if latest is not None else None,
                    "change_1d": change_1d,
                    "history": ordered,
                }
            )
        return entries

    def _get_market_run_series(
        self,
        db: Session,
        market: str,
        latest_run: FeatureRun,
    ) -> list[FeatureRun]:
        normalized_market = market.upper()
        max_runs = max(STATIC_GROUP_HISTORY_RUNS, max(STATIC_GROUP_CHANGE_OFFSETS.values()) + 1)
        published_runs = (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date <= latest_run.as_of_date,
            )
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.published_at.desc(), FeatureRun.id.desc())
            .all()
        )
        market_runs: list[FeatureRun] = []
        seen_dates: set[date] = set()
        for run in published_runs:
            if self._run_market(run) != normalized_market:
                continue
            if run.as_of_date in seen_dates:
                continue
            market_runs.append(run)
            seen_dates.add(run.as_of_date)
            if len(market_runs) >= max_runs:
                break
        return market_runs

    @staticmethod
    def _select_group_history_runs(market_runs: list[FeatureRun]) -> list[FeatureRun]:
        selected_indexes = set(range(min(STATIC_GROUP_HISTORY_RUNS, len(market_runs))))
        selected_indexes.update(
            offset
            for offset in STATIC_GROUP_CHANGE_OFFSETS.values()
            if offset < len(market_runs)
        )
        return [market_runs[index] for index in sorted(selected_indexes)]

    def _compute_group_rankings_from_rows(
        self,
        rows: list[Any],
        *,
        ranking_date: date,
    ) -> list[dict[str, Any]]:
        normalized_rows = [self._extract_group_row_payload(row) for row in rows]
        return self._compute_group_rankings_from_serialized_rows(
            normalized_rows,
            ranking_date=ranking_date,
        )

    def _compute_group_rankings_from_serialized_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        ranking_date: date,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_name = row.get("ibd_industry_group")
            rs_rating = row.get("rs_rating")
            if not group_name or rs_rating is None:
                continue
            grouped[str(group_name)].append(row)

        rankings: list[dict[str, Any]] = []
        ranking_date_str = ranking_date.isoformat()
        for group_name, group_rows in grouped.items():
            rs_values = [float(row["rs_rating"]) for row in group_rows if row.get("rs_rating") is not None]
            if not rs_values:
                continue
            avg_rs = round(sum(rs_values) / len(rs_values), 2)
            median_rs = round(float(pd.Series(rs_values).median()), 2)
            std_dev = round(float(pd.Series(rs_values).std(ddof=0)), 2) if len(rs_values) > 1 else 0.0
            weight_pairs = [
                (
                    float(row.get("market_cap_usd") or row.get("market_cap") or 0),
                    float(row["rs_rating"]),
                )
                for row in group_rows
                if row.get("rs_rating") is not None
            ]
            total_weight = sum(weight for weight, _ in weight_pairs if weight > 0)
            weighted_avg = (
                round(sum(weight * value for weight, value in weight_pairs if weight > 0) / total_weight, 2)
                if total_weight > 0
                else None
            )
            top_row = max(
                group_rows,
                key=lambda row: (
                    row.get("rs_rating") if row.get("rs_rating") is not None else float("-inf"),
                    row.get("composite_score") if row.get("composite_score") is not None else float("-inf"),
                ),
            )
            above_80 = sum(1 for value in rs_values if value >= 80)
            rankings.append(
                {
                    "industry_group": group_name,
                    "date": ranking_date_str,
                    "rank": 0,
                    "avg_rs_rating": avg_rs,
                    "median_rs_rating": median_rs,
                    "weighted_avg_rs_rating": weighted_avg,
                    "rs_std_dev": std_dev,
                    "num_stocks": len(rs_values),
                    "num_stocks_rs_above_80": above_80,
                    "pct_rs_above_80": round((above_80 / len(rs_values)) * 100, 2) if rs_values else None,
                    "top_symbol": top_row.get("symbol"),
                    "top_symbol_name": top_row.get("company_name"),
                    "top_rs_rating": top_row.get("rs_rating"),
                    "rank_change_1w": None,
                    "rank_change_1m": None,
                    "rank_change_3m": None,
                    "rank_change_6m": None,
                }
            )

        rankings.sort(
            key=lambda row: (
                -(row.get("avg_rs_rating") or 0),
                -(row.get("weighted_avg_rs_rating") or 0),
                -(row.get("num_stocks") or 0),
                row["industry_group"],
            )
        )
        for index, row in enumerate(rankings, start=1):
            row["rank"] = index
        return rankings

    @staticmethod
    def _extract_group_row_payload(row: Any) -> dict[str, Any]:
        extended = getattr(row, "extended_fields", {}) or {}
        return {
            "symbol": getattr(row, "symbol", None),
            "company_name": extended.get("company_name"),
            "composite_score": getattr(row, "composite_score", None),
            "current_price": getattr(row, "current_price", None),
            "rs_rating": extended.get("rs_rating"),
            "rs_rating_1m": extended.get("rs_rating_1m"),
            "rs_rating_3m": extended.get("rs_rating_3m"),
            "rs_rating_12m": extended.get("rs_rating_12m"),
            "eps_growth_qq": extended.get("eps_growth_qq"),
            "eps_growth_yy": extended.get("eps_growth_yy"),
            "sales_growth_qq": extended.get("sales_growth_qq"),
            "sales_growth_yy": extended.get("sales_growth_yy"),
            "stage": extended.get("stage"),
            "market_cap": extended.get("market_cap"),
            "market_cap_usd": extended.get("market_cap_usd"),
            "ibd_industry_group": extended.get("ibd_industry_group"),
            "price_sparkline_data": extended.get("price_sparkline_data"),
            "price_trend": extended.get("price_trend"),
            "price_change_1d": extended.get("price_change_1d"),
            "rs_sparkline_data": extended.get("rs_sparkline_data"),
            "rs_trend": extended.get("rs_trend"),
        }

    @staticmethod
    def _group_rank_map(rankings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {row["industry_group"]: row for row in rankings}

    def _apply_group_rank_changes(
        self,
        rankings: list[dict[str, Any]],
        market_runs: list[FeatureRun],
        historical_rankings: dict[int, list[dict[str, Any]]],
    ) -> None:
        for period, offset in STATIC_GROUP_CHANGE_OFFSETS.items():
            key = f"rank_change_{period}"
            if offset >= len(market_runs):
                for ranking in rankings:
                    ranking[key] = None
                continue
            reference_run = market_runs[offset]
            reference_map = self._group_rank_map(historical_rankings.get(reference_run.id, []))
            for ranking in rankings:
                historical = reference_map.get(ranking["industry_group"])
                ranking[key] = (
                    historical["rank"] - ranking["rank"]
                    if historical is not None
                    else None
                )

    def _build_group_details(
        self,
        *,
        rankings: list[dict[str, Any]],
        serialized_rows: list[dict[str, Any]],
        market_runs: list[FeatureRun],
        historical_rankings: dict[int, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        current_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in serialized_rows:
            group_name = row.get("ibd_industry_group")
            if group_name:
                current_rows_by_group[str(group_name)].append(row)

        history_runs = market_runs[:STATIC_GROUP_HISTORY_RUNS]
        ranking_maps = {
            run.id: self._group_rank_map(historical_rankings.get(run.id, []))
            for run in history_runs
        }
        details: dict[str, Any] = {}
        for ranking in rankings:
            group_name = ranking["industry_group"]
            history = []
            for run in history_runs:
                historical = ranking_maps.get(run.id, {}).get(group_name)
                if historical is None:
                    continue
                history.append(
                    HistoricalDataPoint(
                        date=historical["date"],
                        rank=historical["rank"],
                        avg_rs_rating=historical["avg_rs_rating"],
                        num_stocks=historical["num_stocks"],
                    ).model_dump(mode="json")
                )

            stocks = sorted(
                current_rows_by_group.get(group_name, []),
                key=lambda row: (
                    row.get("rs_rating") if row.get("rs_rating") is not None else float("-inf"),
                    row.get("composite_score") if row.get("composite_score") is not None else float("-inf"),
                ),
                reverse=True,
            )
            stock_payload = [
                ConstituentStock(
                    symbol=row["symbol"],
                    company_name=row.get("company_name"),
                    price=row.get("current_price"),
                    rs_rating=row.get("rs_rating"),
                    rs_rating_1m=row.get("rs_rating_1m"),
                    rs_rating_3m=row.get("rs_rating_3m"),
                    rs_rating_12m=row.get("rs_rating_12m"),
                    eps_growth_qq=row.get("eps_growth_qq"),
                    eps_growth_yy=row.get("eps_growth_yy"),
                    sales_growth_qq=row.get("sales_growth_qq"),
                    sales_growth_yy=row.get("sales_growth_yy"),
                    composite_score=row.get("composite_score"),
                    stage=row.get("stage"),
                    price_sparkline_data=row.get("price_sparkline_data"),
                    price_trend=row.get("price_trend"),
                    price_change_1d=row.get("price_change_1d"),
                    rs_sparkline_data=row.get("rs_sparkline_data"),
                    rs_trend=row.get("rs_trend"),
                ).model_dump(mode="json")
                for row in stocks
            ]

            details[group_name] = GroupDetailResponse(
                industry_group=group_name,
                current_rank=ranking["rank"],
                current_avg_rs=ranking["avg_rs_rating"],
                current_median_rs=ranking.get("median_rs_rating"),
                current_weighted_avg_rs=ranking.get("weighted_avg_rs_rating"),
                current_rs_std_dev=ranking.get("rs_std_dev"),
                num_stocks=ranking["num_stocks"],
                pct_rs_above_80=ranking.get("pct_rs_above_80"),
                top_symbol=ranking.get("top_symbol"),
                top_symbol_name=ranking.get("top_symbol_name"),
                top_rs_rating=ranking.get("top_rs_rating"),
                rank_change_1w=ranking.get("rank_change_1w"),
                rank_change_1m=ranking.get("rank_change_1m"),
                rank_change_3m=ranking.get("rank_change_3m"),
                rank_change_6m=ranking.get("rank_change_6m"),
                history=history,
                stocks=stock_payload,
            ).model_dump(mode="json")
        return details

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

    def _get_market_benchmark_history(self, market: str, *, period: str) -> pd.DataFrame | None:
        for candidate in self._benchmark_cache.get_benchmark_candidates(market):
            history = self._get_symbol_price_history(candidate, period=period)
            if history is not None and not history.empty:
                return history
        return None

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
    def _apply_static_default_filters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        min_volume = STATIC_DEFAULT_SCAN_FILTERS.get("minVolume")
        if min_volume is None:
            return list(rows)
        return [
            row
            for row in rows
            if row.get("volume") is not None and row["volume"] >= min_volume
        ]

    def _serialize_chart_bars(self, data) -> list[dict[str, Any]]:
        if data is None or getattr(data, "empty", True):
            return []

        cutoff_date = datetime.utcnow() - timedelta(days=STATIC_CHART_PERIOD_DAYS)
        cutoff_ts = cutoff_date
        if data.index.tz is not None:
            cutoff_ts = cutoff_date.replace(tzinfo=data.index.tz)

        filtered = data[data.index >= cutoff_ts]
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
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _coerce_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)

"""Static site export service for the daily GitHub Pages build."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import logging
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import quote

from sqlalchemy.orm import Session, sessionmaker

from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.models.stock import StockPrice
from app.schemas.groups import GroupRankResponse, GroupRankingsResponse, MoversResponse
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.ui_snapshot_service import UISnapshotService
from app.wiring.bootstrap import (
    get_fundamentals_cache,
    get_group_rank_service,
    get_price_cache,
)


logger = logging.getLogger(__name__)

STATIC_SITE_SCHEMA_VERSION = "static-site-v1"
SCAN_BUNDLE_SCHEMA_VERSION = "static-scan-v1"
CHART_BUNDLE_SCHEMA_VERSION = "static-charts-v1"
SCAN_CHUNK_SIZE = 1000
STATIC_CHART_LIMIT = 200
STATIC_CHART_PERIOD = "6mo"
STATIC_CHART_PERIOD_DAYS = 180
STATIC_CHART_LOOKUP_BATCH_SIZE = 250
STATIC_DEFAULT_SCAN_FILTERS = {"minVolume": 100_000_000}
STATIC_GROUP_DETAIL_HISTORY_DAYS = 100

_DEFAULT_KEY_MARKETS = (
    {"symbol": "SPY", "display_name": "S&P 500 ETF"},
    {"symbol": "QQQ", "display_name": "Nasdaq 100 ETF"},
    {"symbol": "IWM", "display_name": "Russell 2000 ETF"},
    {"symbol": "GLD", "display_name": "Gold ETF"},
    {"symbol": "TLT", "display_name": "20+ Year Treasury ETF"},
)


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

    def export(self, output_dir: Path, *, clean: bool = True) -> StaticSiteExportResult:
        output_dir = Path(output_dir)
        generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        warnings: list[str] = []

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._session_factory() as db:
            latest_run = self._get_latest_published_run(db)
            if latest_run is None:
                raise RuntimeError("No published feature run is available for static-site export")

            scan_rows, filter_options = self._load_scan_export_source(db, latest_run)
            scan_manifest = self._export_scan_bundle(
                db=db,
                output_dir=output_dir,
                generated_at=generated_at,
                run=latest_run,
                rows=scan_rows,
                filter_options=filter_options,
            )
            chart_manifest = self._export_chart_bundle(
                output_dir=output_dir,
                generated_at=generated_at,
                run=latest_run,
                rows=scan_rows,
            )
            breadth_payload = self._build_optional_section_payload(
                section="breadth",
                warnings=warnings,
                generated_at=generated_at,
                expected_as_of_date=latest_run.as_of_date,
                build=lambda: self._build_breadth_payload(
                    generated_at=generated_at,
                    expected_as_of_date=latest_run.as_of_date,
                ),
            )
            groups_payload = self._build_optional_section_payload(
                section="groups",
                warnings=warnings,
                generated_at=generated_at,
                expected_as_of_date=latest_run.as_of_date,
                build=lambda: self._build_groups_payload(
                    db=db,
                    generated_at=generated_at,
                    expected_as_of_date=latest_run.as_of_date,
                ),
            )
            home_payload = self._build_home_payload(
                db=db,
                generated_at=generated_at,
                latest_run=latest_run,
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
        self._write_json(output_dir / "scan" / "manifest.json", scan_manifest)

        skipped_chart_symbols = chart_manifest.get("skipped_symbols") or []
        if skipped_chart_symbols:
            preview = ", ".join(skipped_chart_symbols[:5])
            warnings.append(
                "Static charts skipped "
                f"{len(skipped_chart_symbols)} symbols without cached {STATIC_CHART_PERIOD} price history"
                + (f": {preview}" if preview else "")
            )

        breadth_path = Path("breadth.json")
        groups_path = Path("groups.json")
        home_path = Path("home.json")

        self._write_json(output_dir / breadth_path, breadth_payload)
        self._write_json(output_dir / groups_path, groups_payload)
        self._write_json(output_dir / home_path, home_payload)

        manifest = {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": latest_run.as_of_date.isoformat(),
            "features": {
                "scan": True,
                "breadth": bool(breadth_payload.get("available", True)),
                "groups": bool(groups_payload.get("available", False)),
                "charts": bool(chart_manifest.get("available", False)),
            },
            "pages": {
                "home": {"path": home_path.as_posix()},
                "scan": {"path": Path("scan/manifest.json").as_posix()},
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
            "warnings": warnings,
        }
        self._write_json(output_dir / "manifest.json", manifest)

        return StaticSiteExportResult(
            output_dir=output_dir,
            generated_at=generated_at,
            as_of_date=latest_run.as_of_date.isoformat(),
            warnings=tuple(warnings),
            manifest=manifest,
        )

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

    def _get_latest_published_run(self, db: Session) -> FeatureRun | None:
        pointer = (
            db.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        if pointer is not None:
            run = db.query(FeatureRun).filter(FeatureRun.id == pointer.run_id).first()
            if run is not None and run.status == "published":
                return run

        return (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.published_at.desc(), FeatureRun.id.desc())
            .first()
        )

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
    ) -> dict[str, Any]:
        if rows is None or filter_options is None:
            repo = SqlFeatureStoreRepository(db)
            rows = repo.query_all_as_scan_results(
                run.id,
                FilterSpec(),
                SortSpec(field="composite_score", order=SortOrder.DESC),
                include_sparklines=True,
            )
            filter_options = repo.get_filter_options_for_run(run.id)

        scan_dir = output_dir / "scan"
        chunk_dir = scan_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_refs: list[dict[str, Any]] = []
        serialized_rows = [self._serialize_scan_row(row) for row in rows]
        default_filtered_rows = self._apply_static_default_filters(serialized_rows)
        for index in range(0, len(serialized_rows), SCAN_CHUNK_SIZE):
            chunk_rows = serialized_rows[index:index + SCAN_CHUNK_SIZE]
            chunk_num = (index // SCAN_CHUNK_SIZE) + 1
            rel_path = Path(f"scan/chunks/chunk-{chunk_num:04d}.json")
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
            "chunks": chunk_refs,
            "initial_rows": default_filtered_rows[:50],
            "preview_rows": default_filtered_rows[:10],
        }
        self._write_json(scan_dir / "manifest.json", manifest)
        return manifest

    def _export_chart_bundle(
        self,
        *,
        output_dir: Path,
        generated_at: str,
        run: FeatureRun,
        rows: list[Any],
    ) -> dict[str, Any]:
        chart_dir = output_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        skipped_symbols: list[str] = []
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

                bars = self._serialize_chart_bars(price_data.get(symbol))
                if not bars:
                    skipped_symbols.append(symbol)
                    continue

                rel_path = self._chart_payload_path(symbol)
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

        index_rel_path = Path("charts/index.json")
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
    ) -> dict[str, Any]:
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

    def _build_groups_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        expected_as_of_date: date,
    ) -> dict[str, Any]:
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
                    db, group_name, days=STATIC_GROUP_DETAIL_HISTORY_DAYS,
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

    def _build_home_payload(
        self,
        *,
        db: Session,
        generated_at: str,
        latest_run: FeatureRun,
        scan_manifest: dict[str, Any],
        breadth_payload: dict[str, Any],
        groups_payload: dict[str, Any],
    ) -> dict[str, Any]:
        key_markets = self._build_key_markets(db)
        top_groups = (
            ((groups_payload.get("payload") or {}).get("rankings") or {}).get("rankings") or []
        )[:10]

        breadth_current = ((breadth_payload.get("payload") or {}).get("current")) or {}

        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": latest_run.as_of_date.isoformat(),
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

    def _build_key_markets(self, db: Session) -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        for item in _DEFAULT_KEY_MARKETS:
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

            markets.append(
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
        return markets

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
    def _chart_payload_path(symbol: str) -> Path:
        return Path("charts") / f"{quote(symbol, safe='')}.json"

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

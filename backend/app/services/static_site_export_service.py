"""Static site export service for the daily GitHub Pages build."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import shutil
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.domain.common.query import FilterSpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.models.stock import StockPrice
from app.schemas.scanning import FilterOptionsResponse, ScanResultItem
from app.services.ui_snapshot_service import UISnapshotService


STATIC_SITE_SCHEMA_VERSION = "static-site-v1"
SCAN_BUNDLE_SCHEMA_VERSION = "static-scan-v1"
SCAN_CHUNK_SIZE = 1000

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


class StaticSiteExportService:
    """Generate a static JSON bundle for the read-only frontend."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._ui_snapshot_service = UISnapshotService(session_factory)

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

            scan_manifest = self._export_scan_bundle(
                db=db,
                output_dir=output_dir,
                generated_at=generated_at,
                run=latest_run,
            )
            breadth_payload = self._build_breadth_payload(generated_at=generated_at)
            groups_payload = self._build_groups_payload(generated_at=generated_at)
            themes_index = self._build_themes_payloads(
                output_dir=output_dir,
                generated_at=generated_at,
                warnings=warnings,
            )
            home_payload = self._build_home_payload(
                db=db,
                generated_at=generated_at,
                latest_run=latest_run,
                scan_manifest=scan_manifest,
                breadth_payload=breadth_payload,
                groups_payload=groups_payload,
                themes_index=themes_index,
            )

        breadth_path = Path("breadth.json")
        groups_path = Path("groups.json")
        home_path = Path("home.json")
        themes_index_path = Path("themes/index.json")

        self._write_json(output_dir / breadth_path, breadth_payload)
        self._write_json(output_dir / groups_path, groups_payload)
        self._write_json(output_dir / home_path, home_payload)
        self._write_json(output_dir / themes_index_path, themes_index)

        manifest = {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of_date": latest_run.as_of_date.isoformat(),
            "features": {
                "scan": True,
                "breadth": bool(breadth_payload.get("available", True)),
                "groups": bool(groups_payload.get("available", False)),
                "themes": bool(themes_index.get("available", False)),
            },
            "pages": {
                "home": {"path": home_path.as_posix()},
                "scan": {"path": Path("scan/manifest.json").as_posix()},
                "breadth": {"path": breadth_path.as_posix()},
                "groups": {"path": groups_path.as_posix()},
                "themes": {"path": themes_index_path.as_posix()},
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

    def _export_scan_bundle(
        self,
        *,
        db: Session,
        output_dir: Path,
        generated_at: str,
        run: FeatureRun,
    ) -> dict[str, Any]:
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
            "filter_options": FilterOptionsResponse(
                ibd_industries=list(filter_options.ibd_industries),
                gics_sectors=list(filter_options.gics_sectors),
                ratings=list(filter_options.ratings),
            ).model_dump(mode="json"),
            "chunks": chunk_refs,
            "preview_rows": serialized_rows[:10],
        }
        self._write_json(scan_dir / "manifest.json", manifest)
        return manifest

    def _build_breadth_payload(self, *, generated_at: str) -> dict[str, Any]:
        snapshot = self._ui_snapshot_service.publish_breadth_bootstrap().to_dict()
        payload = snapshot.get("payload", {})
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": True,
            "published_at": _coerce_datetime(snapshot.get("published_at")),
            "source_revision": snapshot.get("source_revision"),
            "payload": payload,
        }

    def _build_groups_payload(self, *, generated_at: str) -> dict[str, Any]:
        snapshot = self._ui_snapshot_service.publish_groups_bootstrap()
        if snapshot is None:
            return {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": generated_at,
                "available": False,
                "message": "No published group rankings are available.",
                "payload": None,
            }
        raw = snapshot.to_dict()
        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": True,
            "published_at": _coerce_datetime(raw.get("published_at")),
            "source_revision": raw.get("source_revision"),
            "payload": raw.get("payload", {}),
        }

    def _build_themes_payloads(
        self,
        *,
        output_dir: Path,
        generated_at: str,
        warnings: list[str],
    ) -> dict[str, Any]:
        themes_dir = output_dir / "themes"
        themes_dir.mkdir(parents=True, exist_ok=True)

        variants: dict[str, dict[str, Any]] = {}
        for pipeline in ("technical", "fundamental"):
            for theme_view in ("grouped", "flat"):
                variant_key = f"{pipeline}:{theme_view}"
                rel_path = Path(f"themes/{pipeline}-{theme_view}.json")
                try:
                    snapshot = self._ui_snapshot_service.publish_themes_bootstrap(
                        pipeline=pipeline,
                        theme_view=theme_view,
                    ).to_dict()
                    payload = {
                        "schema_version": STATIC_SITE_SCHEMA_VERSION,
                        "generated_at": generated_at,
                        "available": True,
                        "published_at": _coerce_datetime(snapshot.get("published_at")),
                        "source_revision": snapshot.get("source_revision"),
                        "payload": snapshot.get("payload", {}),
                    }
                    self._write_json(output_dir / rel_path, payload)
                    preview_rankings = (
                        (((payload.get("payload") or {}).get("rankings") or {}).get("rankings"))
                        or (((payload.get("payload") or {}).get("l1_rankings") or {}).get("rankings"))
                        or []
                    )[:5]
                    variants[variant_key] = {
                        "available": True,
                        "path": rel_path.as_posix(),
                        "preview_rankings": preview_rankings,
                    }
                except Exception as exc:  # noqa: BLE001 - best effort by design
                    warnings.append(f"Themes export failed for {variant_key}: {exc}")
                    variants[variant_key] = {
                        "available": False,
                        "path": rel_path.as_posix(),
                        "error": str(exc),
                    }

        return {
            "schema_version": STATIC_SITE_SCHEMA_VERSION,
            "generated_at": generated_at,
            "available": any(entry.get("available") for entry in variants.values()),
            "variants": variants,
            "warnings": [warning for warning in warnings if warning.startswith("Themes export failed")],
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
        themes_index: dict[str, Any],
    ) -> dict[str, Any]:
        key_markets = self._build_key_markets(db)
        top_groups = (
            ((groups_payload.get("payload") or {}).get("rankings") or {}).get("rankings") or []
        )[:5]
        top_themes = []
        technical_flat = themes_index.get("variants", {}).get("technical:flat", {})
        if technical_flat.get("available"):
            top_themes = technical_flat.get("preview_rankings") or []

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
                "themes_available": themes_index.get("available", False),
            },
            "key_markets": key_markets,
            "scan_summary": {
                "run_id": latest_run.id,
                "rows_total": scan_manifest.get("rows_total", 0),
                "top_results": scan_manifest.get("preview_rows", []),
            },
            "top_groups": top_groups,
            "top_themes": top_themes,
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
            if latest is not None and previous is not None and previous.close:
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

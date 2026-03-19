"""Bulk provider snapshot publishing and snapshot-backed fundamentals hydration."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from ..models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from .finviz_parser import FinvizParser
from .fundamentals_cache_service import FundamentalsCacheService
from .price_cache_service import PriceCacheService
from .technical_calculator_service import TechnicalCalculatorService

logger = logging.getLogger(__name__)


class ProviderSnapshotService:
    """Build, publish, and hydrate provider-backed fundamentals snapshots."""

    SNAPSHOT_KEY_FUNDAMENTALS = "fundamentals_v1"
    CATEGORY_LOADERS = {
        "overview": ("finvizfinance.screener.overview", "Overview"),
        "valuation": ("finvizfinance.screener.valuation", "Valuation"),
        "financial": ("finvizfinance.screener.financial", "Financial"),
        "ownership": ("finvizfinance.screener.ownership", "Ownership"),
    }
    EXCHANGES = ("NYSE", "NASDAQ", "AMEX")
    HYDRATE_CHUNK_SIZE = 200

    def __init__(self) -> None:
        self.parser = FinvizParser()
        self.price_cache = PriceCacheService.get_instance()
        self.fundamentals_cache = FundamentalsCacheService.get_instance()
        self.technical_calc = TechnicalCalculatorService()

    def _load_screener_class(self, category: str):
        module_name, class_name = self.CATEGORY_LOADERS[category]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def _serialize_raw_value(value: Any) -> Any:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, bool)):
            return value
        return str(value).strip()

    def _fetch_category_dataframe(self, category: str, exchange: str) -> pd.DataFrame:
        screener_cls = self._load_screener_class(category)
        screener = screener_cls()
        screener.set_filter(filters_dict={"Exchange": exchange})
        df = screener.screener_view(verbose=0)
        return df if df is not None else pd.DataFrame()

    def _normalize_row(self, raw_row: Dict[str, Any], exchange: str) -> Dict[str, Any]:
        normalized = self.parser.normalize_fundamentals(raw_row)
        normalized.update(self.parser.normalize_quarterly_growth(raw_row))
        normalized.pop("_raw_data", None)
        normalized["exchange"] = exchange
        if raw_row.get("Company"):
            normalized["company_name"] = raw_row.get("Company")
        if raw_row.get("Sector"):
            normalized["sector"] = raw_row.get("Sector")
        if raw_row.get("Industry"):
            normalized["industry"] = raw_row.get("Industry")
        if raw_row.get("Country"):
            normalized["country"] = raw_row.get("Country")
        return {key: value for key, value in normalized.items() if value is not None}

    def _build_snapshot_rows(self, exchange_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        exchanges = [exchange_filter.upper()] if exchange_filter else list(self.EXCHANGES)
        merged_rows: Dict[str, Dict[str, Any]] = {}

        for exchange in exchanges:
            for category in self.CATEGORY_LOADERS:
                df = self._fetch_category_dataframe(category, exchange)
                if df is None or df.empty:
                    logger.warning("Finviz %s snapshot returned no rows for %s", category, exchange)
                    continue

                for _, series in df.iterrows():
                    symbol = str(series.get("Ticker", "")).strip().upper()
                    if not symbol:
                        continue

                    raw_row = {
                        column: self._serialize_raw_value(value)
                        for column, value in series.to_dict().items()
                    }
                    merged = merged_rows.setdefault(
                        symbol,
                        {
                            "symbol": symbol,
                            "exchange": exchange,
                            "normalized_payload": {"symbol": symbol, "exchange": exchange},
                            "raw_payload": {},
                        },
                    )
                    merged["exchange"] = exchange
                    merged["raw_payload"][category] = raw_row
                    merged["normalized_payload"].update(self._normalize_row(raw_row, exchange))

        for symbol, row in merged_rows.items():
            payload_json = json.dumps(row["normalized_payload"], sort_keys=True, default=str)
            row["row_hash"] = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return merged_rows

    def create_snapshot_run(
        self,
        db: Session,
        *,
        run_mode: str,
        snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS,
        exchange_filter: Optional[str] = None,
        publish: bool = False,
    ) -> Dict[str, Any]:
        """Create a preview or publish snapshot run and optionally publish it."""
        source_revision = f"{snapshot_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        run = ProviderSnapshotRun(
            snapshot_key=snapshot_key,
            run_mode=run_mode,
            status="building",
            source_revision=source_revision,
        )
        db.add(run)
        db.flush()

        merged_rows = self._build_snapshot_rows(exchange_filter=exchange_filter)
        active_symbols = {
            row[0]
            for row in db.query(StockUniverse.symbol).filter(
                StockUniverse.status == UNIVERSE_STATUS_ACTIVE
            ).all()
        }
        missing_active = sorted(symbol for symbol in active_symbols if symbol not in merged_rows)
        coverage_stats = {
            "active_symbols": len(active_symbols),
            "snapshot_symbols": len(merged_rows),
            "covered_active_symbols": len(active_symbols.intersection(merged_rows)),
            "missing_active_symbols": len(missing_active),
        }
        parity_stats = {
            "missing_active_symbols": missing_active[:100],
        }

        rows = [
            ProviderSnapshotRow(
                run_id=run.id,
                symbol=symbol,
                exchange=row["exchange"],
                row_hash=row["row_hash"],
                normalized_payload_json=json.dumps(row["normalized_payload"], sort_keys=True, default=str),
                raw_payload_json=json.dumps(row["raw_payload"], sort_keys=True, default=str),
            )
            for symbol, row in merged_rows.items()
        ]
        if rows:
            db.bulk_save_objects(rows)

        run.symbols_total = len(merged_rows)
        run.symbols_published = len(active_symbols.intersection(merged_rows))
        run.coverage_stats_json = json.dumps(coverage_stats, sort_keys=True)
        run.parity_stats_json = json.dumps(parity_stats, sort_keys=True)
        run.status = "preview_ready" if not publish else "published"
        if publish:
            published_at = datetime.utcnow()
            run.published_at = published_at
            pointer = db.query(ProviderSnapshotPointer).filter(
                ProviderSnapshotPointer.snapshot_key == snapshot_key
            ).first()
            if pointer is None:
                db.add(
                    ProviderSnapshotPointer(
                        snapshot_key=snapshot_key,
                        run_id=run.id,
                        updated_at=published_at,
                    )
                )
            else:
                pointer.run_id = run.id
                pointer.updated_at = published_at

        db.commit()
        return {
            "run_id": run.id,
            "source_revision": run.source_revision,
            "coverage": coverage_stats,
            "parity": parity_stats,
        }

    def get_published_run(self, db: Session, snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS) -> Optional[ProviderSnapshotRun]:
        """Return the currently published snapshot run."""
        pointer = db.query(ProviderSnapshotPointer).filter(
            ProviderSnapshotPointer.snapshot_key == snapshot_key
        ).first()
        if pointer is None:
            return None
        return db.query(ProviderSnapshotRun).filter(
            ProviderSnapshotRun.id == pointer.run_id
        ).first()

    def hydrate_published_snapshot(
        self,
        db: Session,
        *,
        snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS,
    ) -> Dict[str, Any]:
        """Hydrate stock_fundamentals/cache from the currently published snapshot."""
        run = self.get_published_run(db, snapshot_key=snapshot_key)
        if run is None:
            raise ValueError(f"No published snapshot for {snapshot_key}")

        rows = db.query(ProviderSnapshotRow).filter(
            ProviderSnapshotRow.run_id == run.id
        ).all()
        if not rows:
            return {"run_id": run.id, "hydrated": 0, "missing": 0}

        active_symbols = {
            row[0]
            for row in db.query(StockUniverse.symbol).filter(
                StockUniverse.status == UNIVERSE_STATUS_ACTIVE
            ).all()
        }
        active_rows = [row for row in rows if row.symbol in active_symbols]
        hydrated = 0
        missing_prices = 0

        for chunk_start in range(0, len(active_rows), self.HYDRATE_CHUNK_SIZE):
            chunk_rows = active_rows[chunk_start:chunk_start + self.HYDRATE_CHUNK_SIZE]
            chunk_symbols = [row.symbol for row in chunk_rows]
            existing_data = self.fundamentals_cache.get_many(chunk_symbols)
            price_data = self.price_cache.get_many(chunk_symbols, period="2y")
            technicals = self.technical_calc.calculate_batch(price_data)
            technicals_refreshed_at = datetime.utcnow().isoformat()

            for row in chunk_rows:
                snapshot_payload = json.loads(row.normalized_payload_json)
                technical_payload = technicals.get(row.symbol, {})
                if technical_payload:
                    snapshot_payload.update(technical_payload)
                    snapshot_payload["technicals_refreshed_at"] = technicals_refreshed_at
                else:
                    missing_prices += 1

                snapshot_payload["finviz_snapshot_revision"] = run.source_revision
                snapshot_payload["finviz_snapshot_at"] = (
                    run.published_at.isoformat() if run.published_at else run.created_at.isoformat()
                )
                snapshot_payload["description"] = (
                    snapshot_payload.get("description_finviz")
                    or snapshot_payload.get("description_yfinance")
                )

                merged_payload = self.fundamentals_cache._merge_fundamentals(
                    snapshot_payload,
                    existing_data.get(row.symbol) or {},
                )
                self.fundamentals_cache.store(
                    row.symbol,
                    merged_payload,
                    data_source="snapshot",
                )
                hydrated += 1

        return {
            "run_id": run.id,
            "snapshot_revision": run.source_revision,
            "hydrated": hydrated,
            "missing_prices": missing_prices,
        }

    def get_snapshot_stats(self, db: Session, snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS) -> Dict[str, Any]:
        """Return published snapshot stats for API/status reporting."""
        run = self.get_published_run(db, snapshot_key=snapshot_key)
        if run is None:
            return {
                "published_snapshot_revision": None,
                "published_snapshot_age_days": None,
                "snapshot_coverage": None,
                "parity_summary": None,
            }

        now = datetime.utcnow()
        published_at = run.published_at or run.created_at
        coverage = json.loads(run.coverage_stats_json) if run.coverage_stats_json else None
        parity = json.loads(run.parity_stats_json) if run.parity_stats_json else None
        return {
            "published_snapshot_revision": run.source_revision,
            "published_snapshot_age_days": (
                (now - published_at.replace(tzinfo=None)).days if published_at else None
            ),
            "snapshot_coverage": coverage,
            "parity_summary": parity,
        }


provider_snapshot_service = ProviderSnapshotService()

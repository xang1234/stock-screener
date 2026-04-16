"""Bulk provider snapshot publishing and snapshot-backed fundamentals hydration."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..config import settings
from ..models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from ..models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from ..utils.symbol_support import is_unsupported_yahoo_price_symbol
from .bulk_data_fetcher import BulkDataFetcher
from .finviz_parser import FinvizParser
from .security_master_service import security_master_resolver
from .technical_calculator_service import TechnicalCalculatorService

if TYPE_CHECKING:
    from app.services.fundamentals_cache_service import FundamentalsCacheService
    from app.services.price_cache_service import PriceCacheService
    from app.services.rate_limiter import RedisRateLimiter

logger = logging.getLogger(__name__)


WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION = "weekly-reference-bundle-v1"
WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION = "weekly-reference-manifest-v1"
WEEKLY_REFERENCE_RELEASE_TAG = "weekly-reference-data"
WEEKLY_REFERENCE_LATEST_MANIFEST_NAME = "weekly-reference-latest.json"
WEEKLY_REFERENCE_MARKETS: tuple[str, ...] = ("US", "HK", "JP", "TW")
WEEKLY_REFERENCE_SNAPSHOT_KEYS: dict[str, str] = {
    "US": "fundamentals_v1_us",
    "HK": "fundamentals_v1_hk",
    "JP": "fundamentals_v1_jp",
    "TW": "fundamentals_v1_tw",
}
WEEKLY_REFERENCE_MARKET_BY_SNAPSHOT_KEY: dict[str, str] = {
    snapshot_key: market for market, snapshot_key in WEEKLY_REFERENCE_SNAPSHOT_KEYS.items()
}


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _deserialize_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class ProviderSnapshotService:
    """Build, publish, and hydrate provider-backed fundamentals snapshots."""

    SNAPSHOT_KEY_FUNDAMENTALS = WEEKLY_REFERENCE_SNAPSHOT_KEYS["US"]
    SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET = WEEKLY_REFERENCE_SNAPSHOT_KEYS
    CATEGORY_LOADERS = {
        "overview": ("finvizfinance.screener.overview", "Overview"),
        "valuation": ("finvizfinance.screener.valuation", "Valuation"),
        "financial": ("finvizfinance.screener.financial", "Financial"),
        "ownership": ("finvizfinance.screener.ownership", "Ownership"),
    }
    EXCHANGES = ("NYSE", "NASDAQ", "AMEX")
    HYDRATE_CHUNK_SIZE = 200
    YAHOO_ONLY_REQUIRED_KEYS = (
        "ipo_date",
        "first_trade_date",
        "eps_growth_qq",
        "sales_growth_qq",
        "eps_growth_yy",
        "sales_growth_yy",
        "recent_quarter_date",
        "previous_quarter_date",
        "eps_5yr_cagr",
        "eps_q1_yoy",
        "eps_q2_yoy",
        "eps_raw_score",
        "eps_years_available",
    )
    YAHOO_UNSUPPORTED_SUFFIXES = ("U", "UN", "UNT", "UNIT", "R", "RT")
    YAHOO_UNSUPPORTED_PREFIXES = ("W", "WS", "WT")
    WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION = WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION
    WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION = WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION
    WEEKLY_REFERENCE_RELEASE_TAG = WEEKLY_REFERENCE_RELEASE_TAG
    WEEKLY_REFERENCE_LATEST_MANIFEST_NAME = WEEKLY_REFERENCE_LATEST_MANIFEST_NAME

    @classmethod
    def snapshot_key_for_market(cls, market: str) -> str:
        normalized_market = str(market or "").strip().upper()
        if normalized_market not in cls.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET:
            raise ValueError(
                f"Unsupported weekly reference market {market!r}. "
                f"Expected one of {sorted(cls.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET)}."
            )
        return cls.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET[normalized_market]

    @classmethod
    def market_for_snapshot_key(cls, snapshot_key: str) -> str:
        normalized_key = str(snapshot_key or "").strip()
        if normalized_key == "fundamentals_v1":
            return "US"
        market = WEEKLY_REFERENCE_MARKET_BY_SNAPSHOT_KEY.get(normalized_key)
        if market is None:
            raise ValueError(f"Unsupported weekly reference snapshot key {snapshot_key!r}")
        return market

    @classmethod
    def weekly_reference_snapshot_keys(cls) -> tuple[str, ...]:
        return tuple(
            cls.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET[market]
            for market in WEEKLY_REFERENCE_MARKETS
        )

    @classmethod
    def weekly_reference_latest_manifest_name_for_market(cls, market: str) -> str:
        return f"weekly-reference-latest-{str(market or '').strip().lower()}.json"

    @staticmethod
    def _infer_market_from_bundle_rows(
        universe_rows: Iterable[Dict[str, Any]],
        snapshot_rows: Iterable[Dict[str, Any]],
    ) -> str | None:
        for row in [*list(universe_rows), *list(snapshot_rows)]:
            identity = security_master_resolver.resolve_identity(
                symbol=str(row.get("symbol") or ""),
                market=row.get("market"),
                exchange=row.get("exchange"),
                currency=row.get("currency"),
                timezone=row.get("timezone"),
                local_code=row.get("local_code"),
            )
            if identity.market:
                return identity.market
        return None

    def __init__(
        self,
        *,
        price_cache: PriceCacheService | None = None,
        fundamentals_cache: FundamentalsCacheService | None = None,
        rate_limiter: RedisRateLimiter | None = None,
    ) -> None:
        if rate_limiter is None:
            from .rate_limiter import RedisRateLimiter

            rate_limiter = RedisRateLimiter()
        if price_cache is None or fundamentals_cache is None:
            from app.database import SessionLocal
            from app.services.redis_pool import get_redis_client
            from app.services.fundamentals_cache_service import FundamentalsCacheService
            from app.services.price_cache_service import PriceCacheService

            redis_client = get_redis_client()
            price_cache = price_cache or PriceCacheService(
                redis_client=redis_client,
                session_factory=SessionLocal,
            )
            fundamentals_cache = fundamentals_cache or FundamentalsCacheService(
                redis_client=redis_client,
                session_factory=SessionLocal,
            )
        self.parser = FinvizParser()
        self.price_cache = price_cache
        self.fundamentals_cache = fundamentals_cache
        self.rate_limiter = rate_limiter
        self.technical_calc = TechnicalCalculatorService()
        self.bulk_fetcher = BulkDataFetcher(rate_limiter=rate_limiter)

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

    def _fetch_category_dataframe(
        self,
        category: str,
        exchange: str,
        *,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        screener_cls = self._load_screener_class(category)
        screener = screener_cls()
        screener.set_filter(filters_dict={"Exchange": exchange})
        df = screener.screener_view(verbose=1 if show_progress else 0)
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

    @classmethod
    def _should_skip_yahoo_price_enrichment(cls, symbol: str) -> bool:
        """Return True for derivative-style suffixes that Yahoo price history often lacks."""
        return is_unsupported_yahoo_price_symbol(symbol)

    def _build_snapshot_rows(
        self,
        exchange_filter: Optional[str] = None,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_finviz_progress: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        exchanges = [exchange_filter.upper()] if exchange_filter else list(self.EXCHANGES)
        merged_rows: Dict[str, Dict[str, Any]] = {}
        total_fetches = len(exchanges) * len(self.CATEGORY_LOADERS)
        completed_fetches = 0

        for exchange in exchanges:
            for category in self.CATEGORY_LOADERS:
                df = self._fetch_category_dataframe(
                    category,
                    exchange,
                    show_progress=show_finviz_progress,
                )
                if df is None or df.empty:
                    logger.warning("Finviz %s snapshot returned no rows for %s", category, exchange)
                else:
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

                completed_fetches += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "snapshot_fetch_complete",
                            "exchange": exchange,
                            "category": category,
                            "completed_fetches": completed_fetches,
                            "total_fetches": total_fetches,
                            "rows": 0 if df is None else len(df),
                            "percent_complete": (
                                round((completed_fetches / total_fetches) * 100, 1)
                                if total_fetches
                                else 100.0
                            ),
                        }
                    )

        for symbol, row in merged_rows.items():
            payload_json = json.dumps(row["normalized_payload"], sort_keys=True, default=str)
            row["row_hash"] = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return merged_rows

    @staticmethod
    def _has_value(value: Any) -> bool:
        return value not in (None, "")

    def _needs_yahoo_hydration(self, payload: Dict[str, Any]) -> bool:
        return any(not self._has_value(payload.get(key)) for key in self.YAHOO_ONLY_REQUIRED_KEYS)

    def build_market_snapshot_row(
        self,
        *,
        market: str,
        symbol: str,
        exchange: str | None,
        normalized_payload: Dict[str, Any],
        raw_payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return a ProviderSnapshotRow-ready payload for a market-scoped bundle."""
        identity = security_master_resolver.resolve_identity(
            symbol=str(symbol or ""),
            market=market,
            exchange=exchange,
            currency=normalized_payload.get("currency"),
            timezone=normalized_payload.get("timezone"),
            local_code=normalized_payload.get("local_code"),
        )
        payload = dict(normalized_payload)
        payload.setdefault("symbol", identity.canonical_symbol)
        payload.setdefault("market", identity.market)
        payload.setdefault("exchange", identity.exchange)
        payload.setdefault("currency", identity.currency)
        payload.setdefault("timezone", identity.timezone)
        payload.setdefault("local_code", identity.local_code)
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        return {
            "symbol": identity.canonical_symbol,
            "exchange": identity.exchange,
            "row_hash": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            "normalized_payload": payload,
            "raw_payload": raw_payload,
        }

    def _coverage_gate(
        self,
        coverage_stats: Dict[str, Any],
    ) -> tuple[bool, list[str]]:
        active_symbols = coverage_stats.get("active_symbols", 0) or 0
        covered_active_symbols = coverage_stats.get("covered_active_symbols", 0) or 0
        missing_active_symbols = coverage_stats.get("missing_active_symbols", 0) or 0
        active_coverage = (
            covered_active_symbols / active_symbols
            if active_symbols > 0 else 1.0
        )
        warnings: list[str] = []

        if active_coverage < settings.provider_snapshot_min_active_coverage:
            warnings.append(
                "Active snapshot coverage "
                f"{active_coverage:.2%} below minimum "
                f"{settings.provider_snapshot_min_active_coverage:.2%}"
            )
        if missing_active_symbols > settings.provider_snapshot_max_missing_active_symbols:
            warnings.append(
                "Missing active symbols "
                f"{missing_active_symbols} above maximum "
                f"{settings.provider_snapshot_max_missing_active_symbols}"
            )
        return (not warnings), warnings

    @staticmethod
    def _replace_snapshot_key_runs(db: Session, *, snapshot_key: str) -> None:
        existing_run_ids = [
            row[0]
            for row in db.query(ProviderSnapshotRun.id)
            .filter(ProviderSnapshotRun.snapshot_key == snapshot_key)
            .all()
        ]
        db.query(ProviderSnapshotPointer).filter(
            ProviderSnapshotPointer.snapshot_key == snapshot_key
        ).delete()
        if existing_run_ids:
            db.query(ProviderSnapshotRow).filter(
                ProviderSnapshotRow.run_id.in_(existing_run_ids)
            ).delete(synchronize_session=False)
            db.query(ProviderSnapshotRun).filter(
                ProviderSnapshotRun.id.in_(existing_run_ids)
            ).delete(synchronize_session=False)

    @staticmethod
    def _replace_market_universe_rows(
        db: Session,
        *,
        market: str,
        rows: Iterable[Dict[str, Any]],
    ) -> int:
        db.query(StockUniverse).filter(StockUniverse.market == market).delete(
            synchronize_session=False
        )
        imported_universe = [
            StockUniverse(**ProviderSnapshotService._deserialize_universe_row(row))
            for row in rows
        ]
        if imported_universe:
            db.bulk_save_objects(imported_universe)
        return len(imported_universe)

    @staticmethod
    def _replace_all_universe_rows(
        db: Session,
        *,
        rows: Iterable[Dict[str, Any]],
    ) -> int:
        db.query(StockUniverse).delete(synchronize_session=False)
        imported_universe = [
            StockUniverse(**ProviderSnapshotService._deserialize_universe_row(row))
            for row in rows
        ]
        if imported_universe:
            db.bulk_save_objects(imported_universe)
        return len(imported_universe)

    def publish_market_snapshot_run(
        self,
        db: Session,
        *,
        snapshot_key: str,
        market: str,
        source_revision: str,
        rows: Iterable[Dict[str, Any]],
        coverage_stats: Dict[str, Any],
        warnings: Optional[list[str]] = None,
        run_mode: str = "publish",
        publish: bool = True,
    ) -> Dict[str, Any]:
        """Publish a pre-built market snapshot using the standard run/pointer tables."""
        normalized_market = str(market or "").strip().upper()
        parity_stats = {
            "market": normalized_market,
            "missing_active_symbols": [],
        }
        coverage_ok, coverage_warnings = self._coverage_gate(coverage_stats)
        all_warnings = list(warnings or [])
        all_warnings.extend(coverage_warnings)

        run = ProviderSnapshotRun(
            snapshot_key=snapshot_key,
            run_mode=run_mode,
            status="building",
            source_revision=source_revision,
        )
        db.add(run)
        db.flush()

        materialized_rows = list(rows)
        if materialized_rows:
            db.bulk_save_objects(
                [
                    ProviderSnapshotRow(
                        run_id=run.id,
                        symbol=row["symbol"],
                        exchange=row.get("exchange"),
                        row_hash=row["row_hash"],
                        normalized_payload_json=json.dumps(
                            row["normalized_payload"], sort_keys=True, default=str
                        ),
                        raw_payload_json=(
                            json.dumps(row["raw_payload"], sort_keys=True, default=str)
                            if row.get("raw_payload") is not None
                            else None
                        ),
                    )
                    for row in materialized_rows
                ]
            )

        run.symbols_total = int(coverage_stats.get("snapshot_symbols", len(materialized_rows)) or 0)
        run.symbols_published = int(
            coverage_stats.get("covered_active_symbols", len(materialized_rows)) or 0
        )
        run.coverage_stats_json = json.dumps(coverage_stats, sort_keys=True)
        run.parity_stats_json = json.dumps(parity_stats, sort_keys=True)
        run.warnings_json = json.dumps(all_warnings, sort_keys=True) if all_warnings else None
        run.status = "preview_ready" if not publish else "published"
        published = False
        if publish and coverage_ok:
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
            published = True
        elif publish and not coverage_ok:
            run.status = "publish_blocked"

        db.commit()
        return {
            "run_id": run.id,
            "source_revision": run.source_revision,
            "coverage": coverage_stats,
            "parity": parity_stats,
            "published": published,
            "warnings": all_warnings,
            "market": normalized_market,
        }

    def _fetch_yahoo_only_fields(self, symbol: str) -> Dict[str, Any]:
        import yfinance as yf

        yahoo_payload: Dict[str, Any] = {}
        now_iso = datetime.utcnow().isoformat()

        try:
            self.rate_limiter.wait("yfinance", min_interval_s=1.0 / settings.yfinance_rate_limit)
            ticker = yf.Ticker(symbol)
        except Exception as exc:
            logger.warning("Failed to initialize Yahoo hydrator for %s: %s", symbol, exc)
            return yahoo_payload

        try:
            quarterly_growth = self.bulk_fetcher._extract_quarterly_growth(ticker)
            yahoo_payload.update(
                {
                    key: value
                    for key, value in quarterly_growth.items()
                    if key != "_raw_data" and self._has_value(value)
                }
            )
            eps_rating = self.bulk_fetcher._extract_eps_rating_data(ticker)
            yahoo_payload.update(
                {key: value for key, value in eps_rating.items() if self._has_value(value)}
            )
            yahoo_payload["yahoo_statements_refreshed_at"] = now_iso
        except Exception as exc:
            logger.warning("Failed Yahoo statement hydration for %s: %s", symbol, exc)

        try:
            info = ticker.info or {}
            first_trade_date_ms = info.get("firstTradeDateMilliseconds")
            if first_trade_date_ms:
                yahoo_payload["first_trade_date_ms"] = first_trade_date_ms
                ipo_date = self.fundamentals_cache._parse_ipo_date(first_trade_date_ms)
                if ipo_date is not None:
                    yahoo_payload["ipo_date"] = ipo_date
                    yahoo_payload["first_trade_date"] = int(
                        datetime.combine(ipo_date, datetime.min.time()).timestamp()
                    )
            if info.get("longBusinessSummary"):
                yahoo_payload["description_yfinance"] = info.get("longBusinessSummary")
            yahoo_payload["yahoo_profile_refreshed_at"] = now_iso
        except Exception as exc:
            logger.warning("Failed Yahoo profile hydration for %s: %s", symbol, exc)

        return yahoo_payload

    def create_snapshot_run(
        self,
        db: Session,
        *,
        run_mode: str,
        snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS,
        market: Optional[str] = None,
        exchange_filter: Optional[str] = None,
        publish: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_finviz_progress: bool = False,
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

        merged_rows = self._build_snapshot_rows(
            exchange_filter=exchange_filter,
            progress_callback=progress_callback,
            show_finviz_progress=show_finviz_progress,
        )
        active_symbols_query = db.query(StockUniverse.symbol).filter(StockUniverse.active_filter())
        if market is not None:
            active_symbols_query = active_symbols_query.filter(
                StockUniverse.market == str(market).strip().upper()
            )
        active_symbols = {row[0] for row in active_symbols_query.all()}
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
        coverage_ok, coverage_warnings = self._coverage_gate(coverage_stats)

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
        run.warnings_json = json.dumps(coverage_warnings, sort_keys=True) if coverage_warnings else None
        run.status = "preview_ready" if not publish else "published"
        published = False
        if publish and coverage_ok:
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
            published = True
        elif publish and not coverage_ok:
            run.status = "publish_blocked"

        db.commit()
        return {
            "run_id": run.id,
            "source_revision": run.source_revision,
            "coverage": coverage_stats,
            "parity": parity_stats,
            "published": published,
            "warnings": coverage_warnings,
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
        allow_yahoo_hydration: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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
                StockUniverse.active_filter()
            ).all()
        }
        active_rows = [row for row in rows if row.symbol in active_symbols]
        hydrated = 0
        missing_prices = 0
        yahoo_hydrated = 0
        missing_yahoo = 0
        skipped_yahoo_price_symbols = 0
        skipped_yahoo_field_symbols = 0
        total_symbols = len(active_rows)
        total_chunks = (total_symbols + self.HYDRATE_CHUNK_SIZE - 1) // self.HYDRATE_CHUNK_SIZE if total_symbols else 0

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "hydrate_start",
                    "total_symbols": total_symbols,
                    "total_chunks": total_chunks,
                    "chunk_size": self.HYDRATE_CHUNK_SIZE,
                }
            )

        for chunk_index, chunk_start in enumerate(range(0, len(active_rows), self.HYDRATE_CHUNK_SIZE), start=1):
            chunk_rows = active_rows[chunk_start:chunk_start + self.HYDRATE_CHUNK_SIZE]
            chunk_symbols = [row.symbol for row in chunk_rows]
            existing_data = self.fundamentals_cache.get_many(chunk_symbols)
            live_price_symbols = [
                symbol
                for symbol in chunk_symbols
                if not self._should_skip_yahoo_price_enrichment(symbol)
            ]
            cached_only_symbols = [
                symbol
                for symbol in chunk_symbols
                if symbol not in live_price_symbols
            ]

            price_data: Dict[str, Any] = {}
            if live_price_symbols:
                price_data.update(self.price_cache.get_many(live_price_symbols, period="2y"))
            if cached_only_symbols:
                cached_only_getter = getattr(self.price_cache, "get_many_cached_only_fresh", None)
                if not callable(cached_only_getter):
                    cached_only_getter = getattr(self.price_cache, "get_many_cached_only", None)
                if callable(cached_only_getter):
                    price_data.update(cached_only_getter(cached_only_symbols, period="2y"))
                else:
                    price_data.update({symbol: None for symbol in cached_only_symbols})
                skipped_yahoo_price_symbols += len(cached_only_symbols)

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
                merged_payload = self.fundamentals_cache._merge_fundamentals(
                    snapshot_payload,
                    existing_data.get(row.symbol) or {},
                )
                if allow_yahoo_hydration and self._needs_yahoo_hydration(merged_payload):
                    yahoo_payload = self._fetch_yahoo_only_fields(row.symbol)
                    if yahoo_payload:
                        merged_payload = self.fundamentals_cache._merge_fundamentals(
                            merged_payload,
                            yahoo_payload,
                        )
                        yahoo_hydrated += 1
                    if self._needs_yahoo_hydration(merged_payload):
                        missing_yahoo += 1
                elif self._needs_yahoo_hydration(merged_payload):
                    missing_yahoo += 1
                merged_payload["description"] = (
                    merged_payload.get("description_finviz")
                    or merged_payload.get("description_yfinance")
                )
                self.fundamentals_cache.store(
                    row.symbol,
                    merged_payload,
                    data_source="snapshot",
                )
                hydrated += 1

            if progress_callback is not None:
                processed_symbols = chunk_start + len(chunk_rows)
                progress_callback(
                    {
                        "stage": "hydrate_chunk_complete",
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "processed_symbols": processed_symbols,
                        "total_symbols": total_symbols,
                        "percent_complete": (
                            round((processed_symbols / total_symbols) * 100, 1)
                            if total_symbols
                            else 100.0
                        ),
                        "chunk_symbols": len(chunk_rows),
                        "live_price_symbols": len(live_price_symbols),
                        "cached_only_symbols": len(cached_only_symbols),
                        "hydrated": hydrated,
                        "missing_prices": missing_prices,
                        "yahoo_hydrated": yahoo_hydrated,
                        "missing_yahoo": missing_yahoo,
                        "skipped_yahoo_price_symbols": skipped_yahoo_price_symbols,
                        "skipped_yahoo_field_symbols": skipped_yahoo_field_symbols,
                    }
                )

        return {
            "run_id": run.id,
            "snapshot_revision": run.source_revision,
            "hydrated": hydrated,
            "missing_prices": missing_prices,
            "yahoo_hydrated": yahoo_hydrated,
            "missing_yahoo": missing_yahoo,
            "skipped_yahoo_price_symbols": skipped_yahoo_price_symbols,
            "skipped_yahoo_field_symbols": skipped_yahoo_field_symbols,
        }

    def hydrate_all_published_snapshots(
        self,
        db: Session,
        *,
        allow_yahoo_hydration: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Hydrate every market-scoped weekly reference snapshot that is currently published."""
        results: Dict[str, Dict[str, Any]] = {}
        for snapshot_key in self.weekly_reference_snapshot_keys():
            if self.get_published_run(db, snapshot_key=snapshot_key) is None:
                continue
            market = self.market_for_snapshot_key(snapshot_key)
            results[market] = self.hydrate_published_snapshot(
                db,
                snapshot_key=snapshot_key,
                allow_yahoo_hydration=allow_yahoo_hydration,
            )
        legacy_snapshot_key = "fundamentals_v1"
        if (
            self.get_published_run(db, snapshot_key=legacy_snapshot_key) is not None
            and "US" not in results
        ):
            results["US"] = self.hydrate_published_snapshot(
                db,
                snapshot_key=legacy_snapshot_key,
                allow_yahoo_hydration=allow_yahoo_hydration,
            )
        return results

    def get_snapshot_stats(self, db: Session, snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS) -> Dict[str, Any]:
        """Return published snapshot stats for API/status reporting."""
        run = self.get_published_run(db, snapshot_key=snapshot_key)
        if run is None and snapshot_key == self.SNAPSHOT_KEY_FUNDAMENTALS:
            run = self.get_published_run(db, snapshot_key="fundamentals_v1")
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

    def export_weekly_reference_bundle(
        self,
        db: Session,
        *,
        output_path: Path,
        bundle_asset_name: str,
        latest_manifest_path: Path | None = None,
        snapshot_key: str = SNAPSHOT_KEY_FUNDAMENTALS,
        market: str | None = None,
    ) -> Dict[str, Any]:
        """Export the current published fundamentals snapshot + active universe bundle."""
        run = self.get_published_run(db, snapshot_key=snapshot_key)
        if run is None:
            raise ValueError(f"No published snapshot for {snapshot_key}")
        bundle_market = str(market or self.market_for_snapshot_key(snapshot_key)).strip().upper()

        coverage = json.loads(run.coverage_stats_json) if run.coverage_stats_json else None
        parity = json.loads(run.parity_stats_json) if run.parity_stats_json else None
        warnings = json.loads(run.warnings_json) if run.warnings_json else []

        active_universe_rows = (
            db.query(StockUniverse)
            .filter(
                StockUniverse.active_filter(),
                StockUniverse.market == bundle_market,
            )
            .order_by(StockUniverse.symbol.asc())
            .all()
        )
        active_symbols = [row.symbol for row in active_universe_rows]
        active_symbol_set = set(active_symbols)
        fundamentals_by_symbol = self.fundamentals_cache.get_many(active_symbols) if active_symbols else {}
        snapshot_rows = (
            db.query(ProviderSnapshotRow)
            .filter(ProviderSnapshotRow.run_id == run.id)
            .order_by(ProviderSnapshotRow.symbol.asc())
            .all()
        )

        bundle_rows: list[dict[str, Any]] = []
        for row in snapshot_rows:
            if row.symbol not in active_symbol_set:
                continue
            normalized_payload = json.loads(row.normalized_payload_json)
            enriched_payload = self.fundamentals_cache._merge_fundamentals(
                normalized_payload,
                fundamentals_by_symbol.get(row.symbol) or {},
            )
            bundle_rows.append(
                {
                    "symbol": row.symbol,
                    "exchange": row.exchange,
                    "row_hash": row.row_hash,
                    "normalized_payload": enriched_payload,
                }
            )

        bundle_payload = {
            "schema_version": self.WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION,
            "market": bundle_market,
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "as_of_date": (
                (run.published_at or run.created_at).date().isoformat()
                if (run.published_at or run.created_at) is not None
                else None
            ),
            "source_revision": run.source_revision,
            "coverage": coverage,
            "warnings": warnings,
            "snapshot": {
                "snapshot_key": run.snapshot_key,
                "run_mode": run.run_mode,
                "status": run.status,
                "source_revision": run.source_revision,
                "created_at": _serialize_datetime(run.created_at),
                "published_at": _serialize_datetime(run.published_at),
                "symbols_total": run.symbols_total,
                "symbols_published": run.symbols_published,
                "coverage_stats": coverage,
                "parity_stats": parity,
                "warnings": warnings,
                "rows": bundle_rows,
            },
            "universe": [self._serialize_universe_row(row) for row in active_universe_rows],
        }
        self._write_bundle_payload(output_path, bundle_payload)

        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": self.WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION,
            "market": bundle_market,
            "generated_at": bundle_payload["generated_at"],
            "as_of_date": bundle_payload["as_of_date"],
            "source_revision": run.source_revision,
            "coverage": coverage,
            "warnings": warnings,
            "bundle_asset_name": bundle_asset_name,
            "sha256": sha256,
        }
        if latest_manifest_path is not None:
            latest_manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        return {
            "bundle_path": str(output_path),
            "manifest_path": str(latest_manifest_path) if latest_manifest_path is not None else None,
            "bundle_asset_name": bundle_asset_name,
            "sha256": sha256,
            "source_revision": run.source_revision,
            "rows": len(bundle_rows),
            "universe_rows": len(active_universe_rows),
            "as_of_date": bundle_payload["as_of_date"],
            "market": bundle_market,
        }

    def import_weekly_reference_bundle(
        self,
        db: Session,
        *,
        input_path: Path,
    ) -> Dict[str, Any]:
        """Import a weekly reference bundle into the local database."""
        payload = self._read_bundle_payload(input_path)
        if payload.get("schema_version") != self.WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported weekly reference bundle schema version: "
                f"{payload.get('schema_version')}"
            )

        snapshot = payload["snapshot"]
        snapshot_key = snapshot["snapshot_key"]
        universe_rows = payload.get("universe", [])
        snapshot_rows = snapshot.get("rows", [])
        inferred_market = self._infer_market_from_bundle_rows(universe_rows, snapshot_rows)
        bundle_market = str(
            payload.get("market") or inferred_market or self.market_for_snapshot_key(snapshot_key)
        ).strip().upper()
        legacy_global_bundle = snapshot_key == "fundamentals_v1" and not payload.get("market")
        coverage = snapshot.get("coverage_stats")
        parity = snapshot.get("parity_stats")
        warnings = snapshot.get("warnings")

        try:
            self._replace_snapshot_key_runs(db, snapshot_key=snapshot_key)
            if legacy_global_bundle:
                imported_universe_count = self._replace_all_universe_rows(
                    db,
                    rows=universe_rows,
                )
            else:
                imported_universe_count = self._replace_market_universe_rows(
                    db,
                    market=bundle_market,
                    rows=universe_rows,
                )

            run = ProviderSnapshotRun(
                snapshot_key=snapshot["snapshot_key"],
                run_mode=snapshot["run_mode"],
                status=snapshot["status"],
                source_revision=snapshot["source_revision"],
                coverage_stats_json=json.dumps(coverage, sort_keys=True) if coverage is not None else None,
                parity_stats_json=json.dumps(parity, sort_keys=True) if parity is not None else None,
                warnings_json=json.dumps(warnings, sort_keys=True) if warnings else None,
                symbols_total=snapshot.get("symbols_total", len(snapshot_rows)),
                symbols_published=snapshot.get("symbols_published", len(snapshot_rows)),
                created_at=_deserialize_datetime(snapshot.get("created_at")),
                published_at=_deserialize_datetime(snapshot.get("published_at")),
            )
            db.add(run)
            db.flush()

            rows = []
            for row in snapshot_rows:
                identity = security_master_resolver.resolve_identity(
                    symbol=str(row.get("symbol") or ""),
                    market=(
                        row.get("market")
                        or row.get("normalized_payload", {}).get("market")
                        or bundle_market
                    ),
                    exchange=row.get("exchange"),
                    currency=row.get("currency"),
                    timezone=row.get("timezone"),
                    local_code=row.get("local_code"),
                )
                normalized_payload = dict(row["normalized_payload"])
                normalized_payload.update(
                    {
                        "symbol": identity.canonical_symbol,
                        "market": identity.market,
                        "exchange": identity.exchange,
                        "currency": identity.currency,
                        "timezone": identity.timezone,
                        "local_code": identity.local_code,
                    }
                )
                rows.append(
                    ProviderSnapshotRow(
                        run_id=run.id,
                        symbol=identity.canonical_symbol,
                        exchange=identity.exchange,
                        row_hash=row["row_hash"],
                        normalized_payload_json=json.dumps(
                            normalized_payload, sort_keys=True, default=str
                        ),
                        raw_payload_json=None,
                    )
                )
            if rows:
                db.bulk_save_objects(rows)

            db.add(
                ProviderSnapshotPointer(
                    snapshot_key=run.snapshot_key,
                    run_id=run.id,
                    updated_at=_deserialize_datetime(snapshot.get("published_at"))
                    or datetime.utcnow(),
                )
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

        return {
            "run_id": run.id,
            "source_revision": run.source_revision,
            "rows": len(rows),
            "universe_rows": imported_universe_count,
            "as_of_date": payload.get("as_of_date"),
            "market": "MULTI" if legacy_global_bundle else bundle_market,
        }

    @staticmethod
    def _write_bundle_payload(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True, default=str)
        else:
            path.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def _read_bundle_payload(path: Path) -> Dict[str, Any]:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _serialize_universe_row(row: StockUniverse) -> Dict[str, Any]:
        return {
            "symbol": row.symbol,
            "name": row.name,
            "market": row.market,
            "exchange": row.exchange,
            "currency": row.currency,
            "timezone": row.timezone,
            "local_code": row.local_code,
            "sector": row.sector,
            "industry": row.industry,
            "market_cap": row.market_cap,
            "is_active": row.is_active,
            "status": row.status,
            "status_reason": row.status_reason,
            "is_sp500": row.is_sp500,
            "source": row.source,
            "added_at": _serialize_datetime(row.added_at),
            "first_seen_at": _serialize_datetime(row.first_seen_at),
            "last_seen_in_source_at": _serialize_datetime(row.last_seen_in_source_at),
            "deactivated_at": _serialize_datetime(row.deactivated_at),
            "consecutive_fetch_failures": row.consecutive_fetch_failures,
            "last_fetch_success_at": _serialize_datetime(row.last_fetch_success_at),
            "last_fetch_failure_at": _serialize_datetime(row.last_fetch_failure_at),
            "updated_at": _serialize_datetime(row.updated_at),
        }

    @staticmethod
    def _deserialize_universe_row(row: Dict[str, Any]) -> Dict[str, Any]:
        identity = security_master_resolver.resolve_identity(
            symbol=str(row.get("symbol") or ""),
            market=row.get("market"),
            exchange=row.get("exchange"),
            currency=row.get("currency"),
            timezone=row.get("timezone"),
            local_code=row.get("local_code"),
        )

        return {
            "symbol": identity.canonical_symbol,
            "name": row.get("name"),
            "market": identity.market,
            "exchange": identity.exchange,
            "currency": identity.currency,
            "timezone": identity.timezone,
            "local_code": identity.local_code,
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "market_cap": row.get("market_cap"),
            "is_active": row.get("is_active", True),
            "status": row.get("status", UNIVERSE_STATUS_ACTIVE),
            "status_reason": row.get("status_reason"),
            "is_sp500": row.get("is_sp500", False),
            "source": row.get("source", "finviz"),
            "added_at": _deserialize_datetime(row.get("added_at")),
            "first_seen_at": _deserialize_datetime(row.get("first_seen_at")),
            "last_seen_in_source_at": _deserialize_datetime(row.get("last_seen_in_source_at")),
            "deactivated_at": _deserialize_datetime(row.get("deactivated_at")),
            "consecutive_fetch_failures": row.get("consecutive_fetch_failures", 0),
            "last_fetch_success_at": _deserialize_datetime(row.get("last_fetch_success_at")),
            "last_fetch_failure_at": _deserialize_datetime(row.get("last_fetch_failure_at")),
            "updated_at": _deserialize_datetime(row.get("updated_at")),
        }

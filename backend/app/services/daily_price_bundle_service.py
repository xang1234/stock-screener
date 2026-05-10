"""Daily market-scoped price bundle export/import and GitHub sync helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..models.app_settings import AppSetting
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from .github_release_sync_service import GitHubReleaseSyncService
from .market_calendar_service import MarketCalendarService


class DailyPriceBundleService:
    """Export, import, and sync durable market-scoped daily price bundles."""

    DAILY_PRICE_BUNDLE_SCHEMA_VERSION = "daily-price-bundle-v1"
    DAILY_PRICE_MANIFEST_SCHEMA_VERSION = "daily-price-manifest-v1"
    DAILY_PRICE_RELEASE_TAG = "daily-price-data"
    DAILY_PRICE_BAR_PERIOD = "2y"
    DAILY_PRICE_SUPPORTED_MARKETS: tuple[str, ...] = ("US", "HK", "IN", "JP", "KR", "TW", "CN", "CA", "DE")
    SYNC_STATE_CATEGORY = "github_sync"

    def __init__(
        self,
        *,
        price_cache=None,
        market_calendar: MarketCalendarService | None = None,
    ) -> None:
        if price_cache is None:
            from app.database import SessionLocal
            from app.services.redis_pool import get_redis_client
            from app.services.price_cache_service import PriceCacheService

            price_cache = PriceCacheService(
                redis_client=get_redis_client(),
                session_factory=SessionLocal,
            )
        self.price_cache = price_cache
        self.market_calendar = market_calendar or MarketCalendarService()

    @classmethod
    def normalize_market(cls, market: str) -> str:
        normalized = str(market or "").strip().upper()
        if normalized not in cls.DAILY_PRICE_SUPPORTED_MARKETS:
            raise ValueError(
                f"Unsupported daily price bundle market {market!r}. "
                f"Expected one of {sorted(cls.DAILY_PRICE_SUPPORTED_MARKETS)}."
            )
        return normalized

    @classmethod
    def latest_manifest_name_for_market(cls, market: str) -> str:
        return f"daily-price-latest-{cls.normalize_market(market).lower()}.json"

    @classmethod
    def sync_state_key(cls, market: str) -> str:
        return f"github_sync.daily_prices.{cls.normalize_market(market).lower()}"

    @staticmethod
    def _write_bundle_payload(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, default=str)
        else:
            path.write_text(
                json.dumps(payload, sort_keys=True, default=str),
                encoding="utf-8",
            )

    @staticmethod
    def _read_bundle_payload(path: Path) -> dict[str, Any]:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_as_of_date(value: str | None) -> date:
        if not value:
            raise ValueError("Daily price bundle manifest is missing as_of_date")
        return date.fromisoformat(str(value))

    def _validate_manifest_freshness(self, manifest: dict[str, Any]) -> tuple[bool, str | None]:
        market = self.normalize_market(str(manifest.get("market") or ""))
        as_of_date = self._parse_as_of_date(str(manifest.get("as_of_date") or ""))
        expected_session = self.market_calendar.last_completed_trading_day(market)
        if str(manifest.get("bar_period") or "") != self.DAILY_PRICE_BAR_PERIOD:
            raise ValueError(
                f"Daily price manifest bar_period must be {self.DAILY_PRICE_BAR_PERIOD!r}"
            )
        if as_of_date != expected_session:
            return (
                True,
                f"Daily price bundle as_of_date {as_of_date.isoformat()} is behind expected "
                f"session {expected_session.isoformat()}",
            )
        max_age_days = max(int(settings.github_daily_price_max_age_days or 0), 0)
        if max_age_days and (date.today() - as_of_date).days > max_age_days:
            return (
                True,
                f"Daily price bundle is older than the configured max age of {max_age_days} day(s)",
            )
        return False, None

    def _build_batch_dataframe(self, prices: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime([row["date"] for row in prices]),
                "Open": [row.get("open") for row in prices],
                "High": [row.get("high") for row in prices],
                "Low": [row.get("low") for row in prices],
                "Close": [row.get("close") for row in prices],
                "Adj Close": [row.get("adj_close") for row in prices],
                "Volume": [row.get("volume") for row in prices],
            }
        )
        frame.set_index("Date", inplace=True)
        return frame

    def get_import_state(self, db: Session, market: str) -> dict[str, Any] | None:
        setting = (
            db.query(AppSetting)
            .filter(AppSetting.key == self.sync_state_key(market))
            .first()
        )
        if setting is None:
            return None
        try:
            return json.loads(setting.value)
        except json.JSONDecodeError:
            return None

    def _upsert_import_state(
        self,
        db: Session,
        *,
        market: str,
        source_revision: str,
        as_of_date: str,
        symbol_count: int,
        bar_period: str,
    ) -> dict[str, Any]:
        payload = {
            "market": self.normalize_market(market),
            "source_revision": source_revision,
            "as_of_date": as_of_date,
            "symbol_count": int(symbol_count),
            "bar_period": bar_period,
            "imported_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        setting = (
            db.query(AppSetting)
            .filter(AppSetting.key == self.sync_state_key(market))
            .first()
        )
        if setting is None:
            setting = AppSetting(
                key=self.sync_state_key(market),
                category=self.SYNC_STATE_CATEGORY,
                description="Latest imported GitHub daily price bundle metadata",
                value=json.dumps(payload, sort_keys=True),
            )
            db.add(setting)
        else:
            setting.value = json.dumps(payload, sort_keys=True)
            setting.category = self.SYNC_STATE_CATEGORY
            setting.description = "Latest imported GitHub daily price bundle metadata"
        db.commit()
        return payload

    def export_daily_price_bundle(
        self,
        db: Session,
        *,
        market: str,
        output_path: Path,
        bundle_asset_name: str,
        latest_manifest_path: Path | None = None,
        as_of_date: date | None = None,
    ) -> dict[str, Any]:
        bundle_market = self.normalize_market(market)
        bundle_as_of_date = as_of_date or self.market_calendar.last_completed_trading_day(bundle_market)
        start_date = bundle_as_of_date - timedelta(days=730)

        active_rows = (
            db.query(StockUniverse)
            .filter(
                StockUniverse.active_filter(),
                StockUniverse.market == bundle_market,
            )
            .order_by(StockUniverse.symbol.asc())
            .all()
        )
        if not active_rows:
            raise ValueError(f"No active universe rows found for market {bundle_market}")

        universe_by_symbol = {row.symbol: row for row in active_rows}
        all_symbols = list(universe_by_symbol)
        price_rows: list[StockPrice] = []
        for chunk_start in range(0, len(all_symbols), 500):
            chunk_symbols = all_symbols[chunk_start:chunk_start + 500]
            price_rows.extend(
                db.query(StockPrice)
                .filter(
                    StockPrice.symbol.in_(chunk_symbols),
                    StockPrice.date >= start_date,
                    StockPrice.date <= bundle_as_of_date,
                )
                .order_by(StockPrice.symbol.asc(), StockPrice.date.asc())
                .all()
            )

        rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for row in price_rows:
            rows_by_symbol.setdefault(row.symbol, []).append(
                {
                    "date": row.date.isoformat(),
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "adj_close": row.adj_close,
                    "volume": row.volume,
                }
            )

        bundle_rows = [
            {
                "symbol": symbol,
                "exchange": universe_by_symbol[symbol].exchange,
                "prices": rows_by_symbol[symbol],
            }
            for symbol in sorted(rows_by_symbol)
            if rows_by_symbol[symbol]
        ]
        if not bundle_rows:
            raise ValueError(f"No {bundle_market} stock_prices rows were available to export")

        source_revision = f"daily_prices_{bundle_market.lower()}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        bundle_payload = {
            "schema_version": self.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
            "market": bundle_market,
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "as_of_date": bundle_as_of_date.isoformat(),
            "source_revision": source_revision,
            "bar_period": self.DAILY_PRICE_BAR_PERIOD,
            "symbol_count": len(bundle_rows),
            "rows": bundle_rows,
        }
        self._write_bundle_payload(output_path, bundle_payload)

        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": self.DAILY_PRICE_MANIFEST_SCHEMA_VERSION,
            "market": bundle_market,
            "generated_at": bundle_payload["generated_at"],
            "as_of_date": bundle_payload["as_of_date"],
            "source_revision": source_revision,
            "bundle_asset_name": bundle_asset_name,
            "sha256": sha256,
            "bar_period": self.DAILY_PRICE_BAR_PERIOD,
            "symbol_count": len(bundle_rows),
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
            "source_revision": source_revision,
            "market": bundle_market,
            "as_of_date": bundle_payload["as_of_date"],
            "bar_period": self.DAILY_PRICE_BAR_PERIOD,
            "symbol_count": len(bundle_rows),
            "rows": sum(len(row["prices"]) for row in bundle_rows),
        }

    def import_daily_price_bundle(
        self,
        db: Session,
        *,
        input_path: Path,
        warm_redis_symbols: int | None = None,
    ) -> dict[str, Any]:
        payload = self._read_bundle_payload(input_path)
        if payload.get("schema_version") != self.DAILY_PRICE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported daily price bundle schema version: "
                f"{payload.get('schema_version')!r}"
            )

        market = self.normalize_market(str(payload.get("market") or ""))
        as_of_date = self._parse_as_of_date(str(payload.get("as_of_date") or ""))
        bar_period = str(payload.get("bar_period") or "")
        if bar_period != self.DAILY_PRICE_BAR_PERIOD:
            raise ValueError(
                f"Unsupported daily price bundle bar_period {bar_period!r}; "
                f"expected {self.DAILY_PRICE_BAR_PERIOD!r}"
            )

        bundle_rows = payload.get("rows") or []
        batch_data: dict[str, pd.DataFrame] = {}
        imported_rows = 0
        for row in bundle_rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            prices = row.get("prices") or []
            if not symbol or not prices:
                continue
            batch_data[symbol] = self._build_batch_dataframe(prices)
            imported_rows += len(prices)

        if batch_data:
            for chunk_start in range(0, len(batch_data), 100):
                chunk_symbols = list(batch_data.keys())[chunk_start:chunk_start + 100]
                self.price_cache._store_batch_in_database(  # noqa: SLF001 - intentional import path reuse
                    {symbol: batch_data[symbol] for symbol in chunk_symbols}
                )

        redis_target = (
            settings.github_daily_price_redis_warm_symbols
            if warm_redis_symbols is None
            else warm_redis_symbols
        )
        redis_warmed_symbols = 0
        # Bundles currently ship 2y bars. Avoid overwriting the standard 5y
        # Redis cache entries with truncated history during import.
        if redis_target and batch_data and bar_period == "5y":
            warm_symbols = [
                row.symbol
                for row in (
                    db.query(StockUniverse)
                    .filter(
                        StockUniverse.market == market,
                        StockUniverse.symbol.in_(list(batch_data)),
                    )
                    .order_by(StockUniverse.market_cap.desc().nullslast())
                    .limit(int(redis_target))
                    .all()
                )
            ]
            if warm_symbols:
                redis_warmed_symbols = self.price_cache.store_batch_in_cache(
                    {symbol: batch_data[symbol] for symbol in warm_symbols},
                    also_store_db=False,
                )

        sync_state = self._upsert_import_state(
            db,
            market=market,
            source_revision=str(payload.get("source_revision") or ""),
            as_of_date=as_of_date.isoformat(),
            symbol_count=int(payload.get("symbol_count") or len(batch_data)),
            bar_period=bar_period,
        )

        return {
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "source_revision": sync_state["source_revision"],
            "bar_period": bar_period,
            "symbol_count": int(payload.get("symbol_count") or len(batch_data)),
            "imported_symbols": len(batch_data),
            "imported_rows": imported_rows,
            "redis_warmed_symbols": redis_warmed_symbols,
        }

    def symbols_missing_as_of(
        self,
        db: Session,
        *,
        symbols: list[str],
        as_of_date: str | date,
    ) -> list[str]:
        if not symbols:
            return []
        expected_date = (
            as_of_date if isinstance(as_of_date, date) else date.fromisoformat(str(as_of_date))
        )
        latest_by_symbol: dict[str, date | None] = {}
        for chunk_start in range(0, len(symbols), 500):
            chunk_symbols = symbols[chunk_start:chunk_start + 500]
            rows = (
                db.query(StockPrice.symbol, func.max(StockPrice.date))
                .filter(StockPrice.symbol.in_(chunk_symbols))
                .group_by(StockPrice.symbol)
                .all()
            )
            for symbol, latest_date in rows:
                latest_by_symbol[str(symbol)] = latest_date
        return [
            symbol for symbol in symbols
            if latest_by_symbol.get(symbol) is None or latest_by_symbol[symbol] < expected_date
        ]

    def sync_from_github(
        self,
        db: Session,
        *,
        market: str,
        warm_redis_symbols: int | None = None,
        allow_stale: bool = False,
        github_sync_service: GitHubReleaseSyncService | None = None,
    ) -> dict[str, Any]:
        normalized_market = str(market or "").strip().upper()
        if normalized_market not in self.DAILY_PRICE_SUPPORTED_MARKETS:
            return {
                "status": "unsupported_market",
                "market": normalized_market,
                "source": "github",
            }

        sync_service = github_sync_service or GitHubReleaseSyncService(
            api_base=settings.github_data_api_base
        )
        import_state = self.get_import_state(db, normalized_market) or {}
        download_dir = Path(tempfile.mkdtemp(prefix=f"daily-price-{normalized_market.lower()}-"))
        sync_result = sync_service.fetch_latest_bundle(
            repository_full_name=settings.github_data_repository,
            release_tag=settings.github_daily_price_release_tag or self.DAILY_PRICE_RELEASE_TAG,
            manifest_asset_name=self.latest_manifest_name_for_market(normalized_market),
            source_mode=settings.market_data_source_mode,
            current_revision=import_state.get("source_revision"),
            expected_manifest_schema=self.DAILY_PRICE_MANIFEST_SCHEMA_VERSION,
            required_manifest_keys=(
                "market",
                "as_of_date",
                "source_revision",
                "bundle_asset_name",
                "sha256",
                "bar_period",
                "symbol_count",
            ),
            stale_validator=self._validate_manifest_freshness,
            allow_stale=allow_stale,
            github_token=settings.github_data_token,
            request_timeout_seconds=settings.github_data_timeout_seconds,
            output_dir=download_dir,
        )
        sync_result["source"] = "github"
        sync_result["market"] = normalized_market
        bundle_path = sync_result.get("bundle_path")
        manifest = sync_result.get("manifest")
        if isinstance(manifest, dict):
            manifest_market_raw = str(manifest.get("market") or "").strip()
            try:
                manifest_market = self.normalize_market(manifest_market_raw)
            except ValueError as exc:
                if bundle_path:
                    Path(str(bundle_path)).unlink(missing_ok=True)
                shutil.rmtree(download_dir, ignore_errors=True)
                return {
                    **sync_result,
                    "status": "invalid_manifest",
                    "error": str(exc),
                }
            if manifest_market != normalized_market:
                if bundle_path:
                    Path(str(bundle_path)).unlink(missing_ok=True)
                shutil.rmtree(download_dir, ignore_errors=True)
                return {
                    **sync_result,
                    "status": "invalid_manifest",
                    "error": (
                        f"Daily price manifest market {manifest_market!r} "
                        f"does not match requested market {normalized_market!r}"
                    ),
                }
            sync_result.setdefault("as_of_date", manifest.get("as_of_date"))
            sync_result.setdefault("bar_period", manifest.get("bar_period"))
            sync_result.setdefault("symbol_count", manifest.get("symbol_count"))

        if sync_result.get("status") != "success":
            shutil.rmtree(download_dir, ignore_errors=True)
            return sync_result

        try:
            import_stats = self.import_daily_price_bundle(
                db,
                input_path=Path(str(bundle_path)),
                warm_redis_symbols=warm_redis_symbols,
            )
        finally:
            if bundle_path:
                Path(str(bundle_path)).unlink(missing_ok=True)
            shutil.rmtree(download_dir, ignore_errors=True)

        return {
            **sync_result,
            **import_stats,
        }

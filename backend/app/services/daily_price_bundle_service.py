"""Daily market-scoped price bundle export/import and GitHub sync helpers."""

from __future__ import annotations

import gzip
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..domain.markets import market_registry
from ..models.app_settings import AppSetting
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from ..utils.file_hashing import sha256_file
from .daily_price_bundle_reader import (
    iter_daily_price_bundle_rows,
    read_daily_price_bundle_metadata,
)
from .github_release_sync_service import GitHubReleaseSyncService
from .market_calendar_service import MarketCalendarService
from .price_row_normalization import stock_price_row_from_ohlcv
from .stock_price_persistence import persist_stock_price_mappings


@dataclass(frozen=True)
class DailyPriceBundleMetadata:
    """Typed metadata contract shared by bundle manifests and streamed payloads."""

    schema_version: str
    market: str
    as_of_date: date
    source_revision: str
    bar_period: str
    symbol_count: int

    @classmethod
    def from_bundle_payload(
        cls,
        payload: dict[str, Any],
        *,
        expected_schema_version: str,
        expected_bar_period: str,
        normalize_market: Callable[[str], str],
    ) -> "DailyPriceBundleMetadata":
        schema_version = str(payload.get("schema_version") or "")
        if schema_version != expected_schema_version:
            raise ValueError(
                "Unsupported daily price bundle schema version: "
                f"{payload.get('schema_version')!r}"
            )

        market = normalize_market(str(payload.get("market") or ""))
        as_of_date = cls._required_date(payload, "as_of_date", source="bundle")
        bar_period = cls._required_text(payload, "bar_period", source="bundle")
        if bar_period != expected_bar_period:
            raise ValueError(
                f"Unsupported daily price bundle bar_period {bar_period!r}; "
                f"expected {expected_bar_period!r}"
            )
        return cls(
            schema_version=schema_version,
            market=market,
            as_of_date=as_of_date,
            source_revision=cls._required_text(
                payload,
                "source_revision",
                source="bundle",
            ),
            bar_period=bar_period,
            symbol_count=cls._required_int(payload, "symbol_count", source="bundle"),
        )

    @classmethod
    def expected_from_manifest(
        cls,
        manifest: dict[str, Any],
        *,
        bundle_schema_version: str,
        expected_bar_period: str,
        normalize_market: Callable[[str], str],
    ) -> "DailyPriceBundleMetadata":
        bar_period = cls._required_text(manifest, "bar_period", source="manifest")
        if bar_period != expected_bar_period:
            raise ValueError(
                f"Daily price manifest bar_period must be {expected_bar_period!r}"
            )
        return cls(
            schema_version=bundle_schema_version,
            market=normalize_market(
                cls._required_text(manifest, "market", source="manifest")
            ),
            as_of_date=cls._required_date(manifest, "as_of_date", source="manifest"),
            source_revision=cls._required_text(
                manifest,
                "source_revision",
                source="manifest",
            ),
            bar_period=bar_period,
            symbol_count=cls._required_int(manifest, "symbol_count", source="manifest"),
        )

    def assert_matches_manifest(self, expected: "DailyPriceBundleMetadata | None") -> None:
        if expected is None:
            return
        comparisons = (
            ("schema_version", self.schema_version, expected.schema_version),
            ("market", self.market, expected.market),
            ("as_of_date", self.as_of_date.isoformat(), expected.as_of_date.isoformat()),
            ("source_revision", self.source_revision, expected.source_revision),
            ("bar_period", self.bar_period, expected.bar_period),
            ("symbol_count", self.symbol_count, expected.symbol_count),
        )
        for key, actual, manifest_value in comparisons:
            if actual != manifest_value:
                raise ValueError(
                    f"Daily price bundle {key} {actual!r} "
                    f"does not match manifest {manifest_value!r}"
                )

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str, *, source: str) -> str:
        value = payload.get(key)
        if value in (None, ""):
            raise ValueError(f"Daily price {source} is missing {key}")
        return str(value)

    @classmethod
    def _required_date(cls, payload: dict[str, Any], key: str, *, source: str) -> date:
        return date.fromisoformat(cls._required_text(payload, key, source=source))

    @classmethod
    def _required_int(cls, payload: dict[str, Any], key: str, *, source: str) -> int:
        raw_value = cls._required_text(payload, key, source=source)
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Daily price {source} {key} must be an integer"
            ) from exc


class DailyPriceBundleService:
    """Export, import, and sync durable market-scoped daily price bundles."""

    DAILY_PRICE_BUNDLE_SCHEMA_VERSION = "daily-price-bundle-v1"
    DAILY_PRICE_MANIFEST_SCHEMA_VERSION = "daily-price-manifest-v1"
    DAILY_PRICE_RELEASE_TAG = "daily-price-data"
    DAILY_PRICE_BAR_PERIOD = "2y"
    DAILY_PRICE_SUPPORTED_MARKETS: tuple[str, ...] = market_registry.supported_market_codes()
    DAILY_PRICE_IMPORT_CHUNK_SIZE = 100
    SYNC_STATE_CATEGORY = "github_sync"

    def __init__(
        self,
        *,
        market_calendar: MarketCalendarService | None = None,
    ) -> None:
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
    def _read_bundle_metadata(path: Path) -> dict[str, Any]:
        return read_daily_price_bundle_metadata(path)

    @staticmethod
    def _iter_bundle_rows(path: Path, *, metadata_out: dict[str, Any] | None = None):
        return iter_daily_price_bundle_rows(path, metadata_out=metadata_out)

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

    @classmethod
    def bundle_metadata_from_payload(
        cls,
        payload: dict[str, Any],
    ) -> DailyPriceBundleMetadata:
        return DailyPriceBundleMetadata.from_bundle_payload(
            payload,
            expected_schema_version=cls.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
            expected_bar_period=cls.DAILY_PRICE_BAR_PERIOD,
            normalize_market=cls.normalize_market,
        )

    @classmethod
    def expected_bundle_metadata_from_manifest(
        cls,
        manifest: dict[str, Any],
    ) -> DailyPriceBundleMetadata:
        return DailyPriceBundleMetadata.expected_from_manifest(
            manifest,
            bundle_schema_version=cls.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
            expected_bar_period=cls.DAILY_PRICE_BAR_PERIOD,
            normalize_market=cls.normalize_market,
        )

    @staticmethod
    def _bundle_price_mapping(price: dict[str, Any]) -> dict[str, Any]:
        return {
            "Open": price.get("open"),
            "High": price.get("high"),
            "Low": price.get("low"),
            "Close": price.get("close"),
            "Adj Close": price.get("adj_close"),
            "Volume": price.get("volume"),
        }

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
        commit: bool = True,
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
        if commit:
            db.commit()
        else:
            db.flush()
        return payload

    def _persist_bundle_price_batch(
        self,
        db: Session,
        batch_rows: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        if not batch_rows:
            return {"inserted": 0, "updated": 0}

        price_rows_by_symbol: dict[str, list[dict[str, Any]]] = {}

        for symbol, prices in batch_rows.items():
            normalized_rows: list[dict[str, Any]] = []
            for price_index, price in enumerate(prices, start=1):
                if not isinstance(price, dict):
                    raise ValueError(
                        f"Daily price bundle prices for {symbol} must contain objects "
                        f"(row {price_index})"
                    )
                row_date = date.fromisoformat(str(price.get("date") or ""))
                price_row = stock_price_row_from_ohlcv(
                    symbol=symbol,
                    row_date=row_date,
                    row=self._bundle_price_mapping(price),
                )
                if price_row is None:
                    raise ValueError(
                        f"Daily price bundle prices for {symbol} contain invalid OHLCV "
                        f"(row {price_index})"
                    )
                normalized_rows.append(price_row)
            if normalized_rows:
                price_rows_by_symbol[symbol] = normalized_rows

        if not price_rows_by_symbol:
            return {"inserted": 0, "updated": 0}

        return persist_stock_price_mappings(
            db,
            price_rows_by_symbol,
            chunk_size=self.DAILY_PRICE_IMPORT_CHUNK_SIZE,
        )

    def export_daily_price_bundle(
        self,
        db: Session,
        *,
        market: str,
        output_path: Path,
        bundle_asset_name: str,
        latest_manifest_path: Path | None = None,
        as_of_date: date | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
        require_complete: bool = False,
        allow_stale_complete: bool = False,
        min_symbol_coverage: float | None = None,
    ) -> dict[str, Any]:
        bundle_market = self.normalize_market(market)
        bundle_as_of_date = as_of_date or self.market_calendar.last_completed_trading_day(bundle_market)
        start_date = bundle_as_of_date - timedelta(days=730)
        coverage_floor = 1.0 if min_symbol_coverage is None else float(min_symbol_coverage)
        if not math.isfinite(coverage_floor) or coverage_floor < 0 or coverage_floor > 1:
            raise ValueError("min_symbol_coverage must be between 0 and 1")
        selected_symbols = None
        if symbols is not None:
            selected_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
            if not selected_symbols:
                raise ValueError("symbols must contain at least one symbol when provided")

        active_query = (
            db.query(StockUniverse)
            .filter(
                StockUniverse.active_filter(),
                StockUniverse.market == bundle_market,
            )
        )
        if selected_symbols is not None:
            active_query = active_query.filter(StockUniverse.symbol.in_(selected_symbols))
        active_rows = active_query.order_by(StockUniverse.symbol.asc()).all()
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
        latest_by_symbol = {
            symbol: date.fromisoformat(rows[-1]["date"])
            for symbol, rows in rows_by_symbol.items()
            if rows
        }

        bundle_rows = [
            {
                "symbol": symbol,
                "exchange": universe_by_symbol[symbol].exchange,
                "prices": rows_by_symbol[symbol],
            }
            for symbol in sorted(rows_by_symbol)
            if rows_by_symbol[symbol]
        ]
        missing_price_symbols = [
            symbol for symbol in all_symbols
            if not rows_by_symbol.get(symbol)
        ]
        stale_symbols = [
            symbol for symbol in all_symbols
            if rows_by_symbol.get(symbol)
            and latest_by_symbol.get(symbol) < bundle_as_of_date
        ]
        missing_symbols = (
            missing_price_symbols
            if allow_stale_complete
            else [*missing_price_symbols, *stale_symbols]
        )
        covered_symbol_count = len(all_symbols) - len(missing_symbols)
        symbol_coverage = (
            covered_symbol_count / len(all_symbols)
            if all_symbols
            else 1.0
        )
        if require_complete and missing_symbols:
            if min_symbol_coverage is None or coverage_floor >= 1.0:
                preview = ", ".join(missing_symbols[:10])
                raise ValueError(
                    f"Missing {len(missing_symbols)} {bundle_market} symbols from daily price bundle "
                    f"as of {bundle_as_of_date.isoformat()}: {preview}"
                )
            if symbol_coverage < coverage_floor:
                preview = ", ".join(missing_symbols[:10])
                raise ValueError(
                    f"Daily price bundle coverage {symbol_coverage:.2%} is below required "
                    f"{coverage_floor:.2%} for {bundle_market} as of "
                    f"{bundle_as_of_date.isoformat()}; missing {len(missing_symbols)} "
                    f"symbols: {preview}"
                )
        if not bundle_rows:
            raise ValueError(f"No {bundle_market} stock_prices rows were available to export")

        source_revision = f"daily_prices_{bundle_market.lower()}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        coverage_summary = {
            "symbol_universe_count": len(all_symbols),
            "covered_symbol_count": covered_symbol_count,
            "symbol_coverage": round(symbol_coverage, 6),
            "min_symbol_coverage": coverage_floor if min_symbol_coverage is not None else None,
        }
        bundle_payload = {
            "schema_version": self.DAILY_PRICE_BUNDLE_SCHEMA_VERSION,
            "market": bundle_market,
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "as_of_date": bundle_as_of_date.isoformat(),
            "source_revision": source_revision,
            "bar_period": self.DAILY_PRICE_BAR_PERIOD,
            "symbol_count": len(bundle_rows),
            "missing_symbol_count": len(missing_price_symbols),
            "stale_symbol_count": len(stale_symbols),
            "allow_stale_complete": bool(allow_stale_complete),
            **coverage_summary,
            "rows": bundle_rows,
        }
        self._write_bundle_payload(output_path, bundle_payload)

        sha256 = sha256_file(output_path)
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
            "missing_symbol_count": len(missing_price_symbols),
            "stale_symbol_count": len(stale_symbols),
            "allow_stale_complete": bool(allow_stale_complete),
            **coverage_summary,
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
            "missing_symbol_count": len(missing_price_symbols),
            "stale_symbol_count": len(stale_symbols),
            "allow_stale_complete": bool(allow_stale_complete),
            **coverage_summary,
            "rows": sum(len(row["prices"]) for row in bundle_rows),
        }

    def import_daily_price_bundle(
        self,
        db: Session,
        *,
        input_path: Path,
        warm_redis_symbols: int | None = None,
        expected_metadata: DailyPriceBundleMetadata | None = None,
    ) -> dict[str, Any]:
        _ = warm_redis_symbols
        payload: dict[str, Any] = {}
        batch_rows: dict[str, list[dict[str, Any]]] = {}
        seen_symbols: set[str] = set()
        imported_symbols = 0
        imported_rows = 0

        def flush_batch() -> None:
            if not batch_rows:
                return
            self._persist_bundle_price_batch(db, dict(batch_rows))
            batch_rows.clear()

        try:
            for row in self._iter_bundle_rows(input_path, metadata_out=payload):
                symbol = str(row.get("symbol") or "").strip().upper()
                prices = row.get("prices")
                if not symbol:
                    raise ValueError("Daily price bundle row is missing symbol")
                if not isinstance(prices, list):
                    raise ValueError(
                        f"Daily price bundle prices for {symbol or '<unknown>'} must be a list"
                    )
                if not prices:
                    raise ValueError(f"Daily price bundle prices for {symbol} must not be empty")
                if symbol in seen_symbols:
                    raise ValueError(f"Daily price bundle contains duplicate symbol {symbol}")
                seen_symbols.add(symbol)
                batch_rows[symbol] = prices
                imported_symbols += 1
                imported_rows += len(prices)
                if len(batch_rows) >= self.DAILY_PRICE_IMPORT_CHUNK_SIZE:
                    flush_batch()
            flush_batch()
            bundle_metadata = self.bundle_metadata_from_payload(payload)
            bundle_metadata.assert_matches_manifest(expected_metadata)
            if imported_symbols != bundle_metadata.symbol_count:
                raise ValueError(
                    "Daily price bundle imported symbol count "
                    f"{imported_symbols} does not match manifest symbol_count "
                    f"{bundle_metadata.symbol_count}"
                )

            sync_state = self._upsert_import_state(
                db,
                market=bundle_metadata.market,
                source_revision=bundle_metadata.source_revision,
                as_of_date=bundle_metadata.as_of_date.isoformat(),
                symbol_count=bundle_metadata.symbol_count,
                bar_period=bundle_metadata.bar_period,
                commit=False,
            )
            db.commit()
        except Exception:
            db.rollback()
            raise

        # Bundles currently ship 2y bars. Avoid overwriting the standard 5y Redis
        # cache entries with truncated history during import.
        redis_warmed_symbols = 0

        return {
            "market": bundle_metadata.market,
            "as_of_date": bundle_metadata.as_of_date.isoformat(),
            "source_revision": sync_state["source_revision"],
            "bar_period": bundle_metadata.bar_period,
            "symbol_count": bundle_metadata.symbol_count,
            "imported_symbols": imported_symbols,
            "imported_rows": imported_rows,
            "redis_warmed_symbols": redis_warmed_symbols,
        }

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
                expected_metadata=(
                    self.expected_bundle_metadata_from_manifest(manifest)
                    if isinstance(manifest, dict)
                    else None
                ),
            )
        finally:
            if bundle_path:
                Path(str(bundle_path)).unlink(missing_ok=True)
            shutil.rmtree(download_dir, ignore_errors=True)

        return {
            **sync_result,
            **import_stats,
        }

"""Daily market-scoped price bundle export/import and GitHub sync helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
from sqlalchemy.orm import Session

from ..config import settings
from ..domain.markets import market_registry
from ..models.app_settings import AppSetting
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from .github_release_sync_service import GitHubReleaseSyncService
from .market_calendar_service import MarketCalendarService


class _StreamingJsonReader:
    """Incremental JSON decoder for bundle files with a very large rows array."""

    _CHUNK_SIZE = 1024 * 1024

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._pos = 0
        self._eof = False

    def _compact(self) -> None:
        if self._pos > self._CHUNK_SIZE or self._pos > len(self._buffer) // 2:
            self._buffer = self._buffer[self._pos:]
            self._pos = 0

    def _read_more(self) -> None:
        self._compact()
        chunk = self._handle.read(self._CHUNK_SIZE)
        if chunk == "":
            self._eof = True
            return
        self._buffer += chunk

    def _ensure_char(self) -> str:
        while self._pos >= len(self._buffer):
            if self._eof:
                raise json.JSONDecodeError(
                    "Unexpected end of daily price bundle",
                    self._buffer,
                    self._pos,
                )
            self._read_more()
        return self._buffer[self._pos]

    def skip_whitespace(self) -> None:
        while True:
            char = self._ensure_char()
            if char not in " \t\r\n":
                return
            self._pos += 1

    def expect(self, expected: str) -> None:
        self.skip_whitespace()
        actual = self._ensure_char()
        if actual != expected:
            raise json.JSONDecodeError(
                f"Expected {expected!r}, got {actual!r}",
                self._buffer,
                self._pos,
            )
        self._pos += 1
        self._compact()

    def consume_if(self, expected: str) -> bool:
        self.skip_whitespace()
        if self._ensure_char() != expected:
            return False
        self._pos += 1
        self._compact()
        return True

    def decode_value(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._pos)
            except json.JSONDecodeError:
                if self._eof:
                    raise
                self._read_more()
                continue
            self._pos = end
            self._compact()
            return value

    def skip_value(self) -> None:
        self.skip_whitespace()
        char = self._ensure_char()
        if char in "[{":
            self._skip_compound_value()
            return
        self.decode_value()

    def _skip_compound_value(self) -> None:
        depth = 0
        in_string = False
        escaped = False
        while True:
            char = self._ensure_char()
            self._pos += 1
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                    if depth == 0:
                        self._compact()
                        return

    def iter_array(self) -> Iterator[Any]:
        self.expect("[")
        if self.consume_if("]"):
            return
        while True:
            yield self.decode_value()
            if self.consume_if(","):
                continue
            self.expect("]")
            return


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
    def _open_bundle_text(path: Path) -> TextIO:
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("rt", encoding="utf-8")

    def _read_bundle_metadata(self, path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        with self._open_bundle_text(path) as handle:
            reader = _StreamingJsonReader(handle)
            reader.expect("{")
            if reader.consume_if("}"):
                return metadata
            while True:
                key = reader.decode_value()
                if not isinstance(key, str):
                    raise ValueError("Daily price bundle keys must be strings")
                reader.expect(":")
                if key == "rows":
                    reader.skip_value()
                else:
                    metadata[key] = reader.decode_value()
                if reader.consume_if(","):
                    continue
                reader.expect("}")
                return metadata

    def _iter_bundle_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        with self._open_bundle_text(path) as handle:
            reader = _StreamingJsonReader(handle)
            reader.expect("{")
            if reader.consume_if("}"):
                return
            while True:
                key = reader.decode_value()
                if not isinstance(key, str):
                    raise ValueError("Daily price bundle keys must be strings")
                reader.expect(":")
                if key == "rows":
                    for row in reader.iter_array():
                        if isinstance(row, dict):
                            yield row
                    if reader.consume_if(","):
                        continue
                    reader.expect("}")
                    return
                reader.skip_value()
                if reader.consume_if(","):
                    continue
                reader.expect("}")
                return

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
    ) -> dict[str, Any]:
        payload = self._read_bundle_metadata(input_path)
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

        batch_data: dict[str, pd.DataFrame] = {}
        redis_batch_data: dict[str, pd.DataFrame] = {}
        imported_symbols = 0
        imported_rows = 0

        def flush_batch() -> None:
            if not batch_data:
                return
            # Reuse the cache service's established DB batch writer without
            # keeping the whole bundle resident in memory.
            self.price_cache._store_batch_in_database(dict(batch_data))  # noqa: SLF001
            batch_data.clear()

        for row in self._iter_bundle_rows(input_path):
            symbol = str(row.get("symbol") or "").strip().upper()
            prices = row.get("prices") or []
            if not symbol or not prices:
                continue
            frame = self._build_batch_dataframe(prices)
            batch_data[symbol] = frame
            if bar_period == "5y":
                redis_batch_data[symbol] = frame
            imported_symbols += 1
            imported_rows += len(prices)
            if len(batch_data) >= self.DAILY_PRICE_IMPORT_CHUNK_SIZE:
                flush_batch()
        flush_batch()

        redis_target = (
            settings.github_daily_price_redis_warm_symbols
            if warm_redis_symbols is None
            else warm_redis_symbols
        )
        redis_warmed_symbols = 0
        # Bundles currently ship 2y bars. Avoid overwriting the standard 5y
        # Redis cache entries with truncated history during import.
        if redis_target and redis_batch_data and bar_period == "5y":
            warm_symbols = [
                row.symbol
                for row in (
                    db.query(StockUniverse)
                    .filter(
                        StockUniverse.market == market,
                        StockUniverse.symbol.in_(list(redis_batch_data)),
                    )
                    .order_by(StockUniverse.market_cap.desc().nullslast())
                    .limit(int(redis_target))
                    .all()
                )
            ]
            if warm_symbols:
                redis_warmed_symbols = self.price_cache.store_batch_in_cache(
                    {symbol: redis_batch_data[symbol] for symbol in warm_symbols},
                    also_store_db=False,
                )

        sync_state = self._upsert_import_state(
            db,
            market=market,
            source_revision=str(payload.get("source_revision") or ""),
            as_of_date=as_of_date.isoformat(),
            symbol_count=int(payload.get("symbol_count") or imported_symbols),
            bar_period=bar_period,
        )

        return {
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "source_revision": sync_state["source_revision"],
            "bar_period": bar_period,
            "symbol_count": int(payload.get("symbol_count") or imported_symbols),
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
            )
        finally:
            if bundle_path:
                Path(str(bundle_path)).unlink(missing_ok=True)
            shutil.rmtree(download_dir, ignore_errors=True)

        return {
            **sync_result,
            **import_stats,
        }

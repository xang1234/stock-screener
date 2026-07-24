"""Bulk provider snapshot publishing and snapshot-backed fundamentals hydration."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import logging
import math
import shutil
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Optional
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from finvizfinance.constants import NUMBER_COL
from finvizfinance.util import number_covert, progress_bar, web_scrap
from sqlalchemy.orm import Session

from ..config import settings
from ..domain.markets import market_registry
from ..domain.providers.price_symbol_support import is_unsupported_yahoo_price_symbol
from ..models.provider_snapshot import (
    ProviderSnapshotPointer,
    ProviderSnapshotRow,
    ProviderSnapshotRun,
)
from ..models.stock_universe import (
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
    StockUniverse,
    StockUniverseStatusEvent,
)
from .bulk_data_fetcher import BulkDataFetcher
from .finviz_parser import FinvizParser
from .github_release_sync_service import GitHubReleaseSyncService
from .security_master_service import security_master_resolver
from .technical_calculator_service import TechnicalCalculatorService
from .weekly_reference_github_sync import (
    WEEKLY_REFERENCE_LEGACY_MANIFEST_NAME,
    WEEKLY_REFERENCE_MANIFEST_SCHEMA_VERSION,
    WEEKLY_REFERENCE_RELEASE_TAG,
    fetch_weekly_reference_bundle,
    weekly_reference_manifest_name,
)

if TYPE_CHECKING:
    from app.services.fundamentals_cache_service import FundamentalsCacheService
    from app.services.price_cache_service import PriceCacheService
    from app.services.rate_limiter import RedisRateLimiter

logger = logging.getLogger(__name__)


WEEKLY_REFERENCE_BUNDLE_SCHEMA_VERSION = "weekly-reference-bundle-v1"
WEEKLY_REFERENCE_LATEST_MANIFEST_NAME = WEEKLY_REFERENCE_LEGACY_MANIFEST_NAME
WEEKLY_REFERENCE_MARKETS: tuple[str, ...] = market_registry.supported_market_codes()
WEEKLY_REFERENCE_SNAPSHOT_KEYS: dict[str, str] = {
    market: f"fundamentals_v1_{market.lower()}" for market in WEEKLY_REFERENCE_MARKETS
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


def _as_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite_or_none(value: Any) -> Any:
    """Return ``None`` when ``value`` is NaN/inf; otherwise pass through.
    Yahoo ``.info`` can return ``float('nan')`` for delisted or illiquid
    tickers, and NaN survives ``is not None`` checks but later breaks FX
    normalization and JSON serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        # Market-state fields Yahoo can backfill when Finviz omits them.
        # Without these, a Finviz snapshot missing only market_cap would skip
        # hydration entirely and the Yahoo fallback would never run.
        "market_cap",
        "shares_outstanding",
    )
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
        return weekly_reference_manifest_name(market)

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
        screener.request_params["o"] = "ticker"

        soup = web_scrap(screener.url, screener.request_params)
        page_count = self._finviz_page_count(soup)
        if page_count == 0:
            return pd.DataFrame()

        df = self._parse_finviz_screener_table(soup)
        for page_index in range(1, page_count):
            sleep(1)
            if show_progress:
                progress_bar(page_index, page_count)
            screener.request_params["r"] = page_index * screener.size + 1
            soup = web_scrap(screener.url, screener.request_params)
            page_df = self._parse_finviz_screener_table(soup)
            df = pd.concat([df, page_df], ignore_index=True)
        return df

    @staticmethod
    def _finviz_page_count(soup: Any) -> int:
        page_select = soup.find(id="pageSelect")
        if page_select is None:
            return 0
        return len(page_select.find_all("option"))

    @staticmethod
    def _extract_finviz_ticker(cell: Any) -> str:
        data_ticker = str(cell.get("data-boxover-ticker") or "").strip()
        if data_ticker:
            return data_ticker

        tab_link = cell.find("a", class_=lambda value: value and "tab-link" in value)
        if tab_link is not None:
            tab_text = tab_link.get_text(strip=True)
            if tab_text:
                return tab_text

        for link in cell.find_all("a", href=True):
            ticker_values = parse_qs(urlparse(link["href"]).query).get("t")
            if ticker_values and ticker_values[0]:
                return ticker_values[0]

        return cell.get_text(strip=True)

    @classmethod
    def _parse_finviz_screener_table(cls, soup: Any) -> pd.DataFrame:
        table = soup.find("table", class_="screener_table")
        if table is None:
            return pd.DataFrame()

        rows = table.find_all("tr")
        if not rows:
            return pd.DataFrame()

        headers = [header.get_text(strip=True) for header in rows[0].find_all("th")][1:]
        frame: list[dict[str, Any]] = []
        for row in rows[1:]:
            cells = row.find_all("td")[1:]
            if not cells:
                continue
            if len(cells) != len(headers):
                logger.warning(
                    "Skipping finviz snapshot row with %d cells (expected %d headers)",
                    len(cells),
                    len(headers),
                )
                continue

            payload: dict[str, Any] = {}
            for header, cell in zip(headers, cells, strict=True):
                if header == "Ticker":
                    payload[header] = cls._extract_finviz_ticker(cell)
                elif header in NUMBER_COL:
                    payload[header] = number_covert(cell.get_text(strip=True))
                else:
                    payload[header] = cell.get_text(strip=True)
            frame.append(payload)
        return pd.DataFrame(frame)

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
        """Return a ProviderSnapshotRow-ready payload for a market-scoped bundle.

        The incoming ``symbol`` is already the canonical symbol (it is read from
        ``StockUniverse.symbol``, which was canonicalized at ingest). We therefore
        trust it as-is — applying only light normalization — and must NOT re-derive
        it from ``(symbol, exchange)``. Re-canonicalizing here is unsafe: if a row's
        stored exchange has drifted away from its explicit suffix (e.g. a TPEx
        ``8906.TWO`` row carrying ``exchange="TWSE"``), ``resolve_identity`` would
        rewrite it to ``8906.TW`` and collide with the genuine TWSE sibling, crashing
        the whole market build on ``uq_provider_snapshot_row_run_symbol``. ``identity``
        is still resolved, but only to fill missing payload metadata defaults.
        """
        identity = security_master_resolver.resolve_identity(
            symbol=str(symbol or ""),
            market=market,
            exchange=exchange,
            currency=normalized_payload.get("currency"),
            timezone=normalized_payload.get("timezone"),
            local_code=normalized_payload.get("local_code"),
        )
        canonical_symbol = identity.normalized_symbol
        payload = dict(normalized_payload)
        payload.setdefault("symbol", canonical_symbol)
        payload.setdefault("market", identity.market)
        payload.setdefault("exchange", identity.exchange)
        payload.setdefault("currency", identity.currency)
        payload.setdefault("timezone", identity.timezone)
        payload.setdefault("local_code", identity.local_code)
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        return {
            "symbol": canonical_symbol,
            "exchange": identity.exchange,
            "row_hash": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            "normalized_payload": payload,
            "raw_payload": raw_payload,
        }

    def _coverage_gate(
        self,
        coverage_stats: Dict[str, Any],
        *,
        market: str | None = None,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        active_symbols = coverage_stats.get("active_symbols", 0) or 0
        covered_active_symbols = coverage_stats.get("covered_active_symbols", 0) or 0
        missing_active_symbols = coverage_stats.get("missing_active_symbols", 0) or 0
        active_coverage = (
            covered_active_symbols / active_symbols
            if active_symbols > 0 else 1.0
        )
        missing_ratio = (
            missing_active_symbols / active_symbols
            if active_symbols > 0 else 0.0
        )
        normalized_market = str(market or "US").strip().upper() or "US"
        if normalized_market not in WEEKLY_REFERENCE_MARKETS:
            normalized_market = "US"
        min_active_coverage = getattr(
            settings,
            f"provider_snapshot_min_active_coverage_{normalized_market.lower()}",
        )
        max_missing_ratio = getattr(
            settings,
            f"provider_snapshot_max_missing_ratio_{normalized_market.lower()}",
        )
        warnings: list[str] = []

        if active_coverage < min_active_coverage:
            warnings.append(
                "Active snapshot coverage "
                f"{active_coverage:.2%} below minimum "
                f"{min_active_coverage:.2%}"
            )
        if missing_ratio > max_missing_ratio:
            warnings.append(
                "Missing active symbol ratio "
                f"{missing_ratio:.2%} above maximum "
                f"{max_missing_ratio:.2%}"
            )
        return (
            not warnings,
            warnings,
            {
                "market": normalized_market,
                "min_active_coverage": min_active_coverage,
                "max_missing_ratio": max_missing_ratio,
                "active_coverage": active_coverage,
                "missing_ratio": missing_ratio,
            },
        )

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
    def _deserialize_universe_rows(rows: Iterable[Dict[str, Any]]) -> list[StockUniverse]:
        """Deserialize bundle universe rows into StockUniverse objects, collapsing
        any that canonicalize to the same symbol.

        ``_deserialize_universe_row`` re-canonicalizes each row via the exchange
        (intentionally — it rewrites bare/board-mismatched symbols). A bundle can
        therefore contain two rows that resolve to the *same* canonical symbol: e.g.
        a phantom TW ``.TWO`` copy of a ``.TW`` security whose stored exchange is the
        TWSE ``XTAI``, which collapses ``.TWO`` -> ``.TW``. Inserting both would hit
        the ``StockUniverse.symbol`` unique index, so we collapse by canonical symbol
        (last write wins) and log the collision rather than crash the whole import.
        """
        deduped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            deserialized = ProviderSnapshotService._deserialize_universe_row(row)
            symbol = deserialized["symbol"]
            if symbol in deduped:
                logger.warning(
                    "Collapsing duplicate universe row: raw=%r canonicalized to %s "
                    "(already seen); keeping last occurrence.",
                    row.get("symbol"),
                    symbol,
                )
            deduped[symbol] = deserialized
        return [StockUniverse(**row) for row in deduped.values()]

    @staticmethod
    def _deserialize_snapshot_rows(
        snapshot_rows: Iterable[Dict[str, Any]],
        *,
        run_id: int,
        bundle_market: str,
    ) -> "dict[str, tuple[ProviderSnapshotRow, Dict[str, Any]]]":
        """Build ProviderSnapshotRow objects + their hydration payloads, collapsing
        rows that re-canonicalize to the same symbol.

        Snapshot twin of :meth:`_deserialize_universe_rows`: the same exchange-driven
        canonicalization that collapses a phantom TW ``.TWO``/``.TW`` pair onto one
        symbol would otherwise violate ``uq_provider_snapshot_row_run_symbol``. Keyed
        by canonical symbol so the row and its payload stay in lockstep; last write
        wins, collisions are logged.
        """
        deduped: "dict[str, tuple[ProviderSnapshotRow, Dict[str, Any]]]" = {}
        for row in snapshot_rows:
            # Snapshot rows carry currency/timezone only inside normalized_payload
            # (the export has no top-level keys for them), so read them from there —
            # otherwise resolve_identity falls back to the market default and the
            # update() below silently flattens a non-default currency/timezone.
            # local_code is intentionally left to derive from the symbol: it feeds
            # canonicalization, and the symbol stem is the authoritative source.
            payload = row.get("normalized_payload", {})
            identity = security_master_resolver.resolve_identity(
                symbol=str(row.get("symbol") or ""),
                market=row.get("market") or payload.get("market") or bundle_market,
                exchange=row.get("exchange"),
                currency=row.get("currency") or payload.get("currency"),
                timezone=row.get("timezone") or payload.get("timezone"),
                local_code=row.get("local_code"),
            )
            canonical_symbol = identity.canonical_symbol
            normalized_payload = dict(row["normalized_payload"])
            normalized_payload.update(
                {
                    "symbol": canonical_symbol,
                    "market": identity.market,
                    "exchange": identity.exchange,
                    "currency": identity.currency,
                    "timezone": identity.timezone,
                    "local_code": identity.local_code,
                }
            )
            if canonical_symbol in deduped:
                logger.warning(
                    "Collapsing duplicate snapshot row: raw=%r canonicalized to %s "
                    "(already seen); keeping last occurrence.",
                    row.get("symbol"),
                    canonical_symbol,
                )
            snapshot_row = ProviderSnapshotRow(
                run_id=run_id,
                symbol=canonical_symbol,
                exchange=identity.exchange,
                row_hash=row["row_hash"],
                normalized_payload_json=json.dumps(
                    normalized_payload, sort_keys=True, default=str
                ),
                raw_payload_json=None,
            )
            deduped[canonical_symbol] = (snapshot_row, normalized_payload)
        return deduped

    @staticmethod
    def _replace_market_universe_rows(
        db: Session,
        *,
        market: str,
        rows: Iterable[Dict[str, Any]],
        lifecycle_event_baseline_at: datetime | None = None,
        lifecycle_event_source_revision: str | None = None,
    ) -> int:
        db.query(StockUniverse).filter(StockUniverse.market == market).delete(
            synchronize_session=False
        )
        return ProviderSnapshotService._insert_universe_rows_with_lifecycle(
            db,
            rows=rows,
            lifecycle_event_baseline_at=lifecycle_event_baseline_at,
            lifecycle_event_source_revision=lifecycle_event_source_revision,
        )

    @staticmethod
    def _replace_all_universe_rows(
        db: Session,
        *,
        rows: Iterable[Dict[str, Any]],
        lifecycle_event_baseline_at: datetime | None = None,
        lifecycle_event_source_revision: str | None = None,
    ) -> int:
        db.query(StockUniverse).delete(synchronize_session=False)
        return ProviderSnapshotService._insert_universe_rows_with_lifecycle(
            db,
            rows=rows,
            lifecycle_event_baseline_at=lifecycle_event_baseline_at,
            lifecycle_event_source_revision=lifecycle_event_source_revision,
        )

    @staticmethod
    def _insert_universe_rows_with_lifecycle(
        db: Session,
        *,
        rows: Iterable[Dict[str, Any]],
        lifecycle_event_baseline_at: datetime | None = None,
        lifecycle_event_source_revision: str | None = None,
    ) -> int:
        imported_universe = ProviderSnapshotService._deserialize_universe_rows(rows)
        ProviderSnapshotService._prepare_imported_universe_lifecycle_rows(
            imported_universe,
            baseline_at=lifecycle_event_baseline_at,
        )
        if imported_universe:
            db.bulk_save_objects(imported_universe)
            ProviderSnapshotService._seed_imported_universe_status_events(
                db,
                rows=imported_universe,
                baseline_at=lifecycle_event_baseline_at,
                source_revision=lifecycle_event_source_revision,
            )
        return len(imported_universe)

    @staticmethod
    def _weekly_reference_lifecycle_baseline_at(
        payload: Dict[str, Any],
        snapshot: Dict[str, Any],
        *,
        market: str | None = None,
    ) -> datetime:
        raw_as_of_date = payload.get("as_of_date")
        if raw_as_of_date:
            try:
                as_of_date = date.fromisoformat(str(raw_as_of_date)[:10])
                timezone_name = None
                raw_market = market or payload.get("market")
                if raw_market:
                    try:
                        timezone_name = market_registry.profile(
                            str(raw_market).strip().upper()
                        ).timezone_name
                    except ValueError:
                        timezone_name = None
                if timezone_name:
                    try:
                        return datetime.combine(
                            as_of_date,
                            time.min,
                            tzinfo=ZoneInfo(timezone_name),
                        ).astimezone(timezone.utc)
                    except ZoneInfoNotFoundError:
                        pass
                return datetime.combine(as_of_date, time.min, tzinfo=timezone.utc)
            except ValueError:
                logger.warning(
                    "Unable to parse weekly reference as_of_date %r for lifecycle seed",
                    raw_as_of_date,
                )

        for raw_timestamp in (
            snapshot.get("published_at"),
            snapshot.get("created_at"),
            payload.get("generated_at"),
        ):
            timestamp = _deserialize_datetime(raw_timestamp)
            if timestamp is not None:
                return _as_utc_datetime(timestamp)

        return datetime.now(timezone.utc)

    @staticmethod
    def _prepare_imported_universe_lifecycle_rows(
        rows: Iterable[StockUniverse],
        *,
        baseline_at: datetime | None,
    ) -> None:
        if baseline_at is None:
            return
        for row in rows:
            if row.first_seen_at is None:
                row.first_seen_at = baseline_at
            if row.added_at is None:
                row.added_at = row.first_seen_at
            if row.is_active and row.last_seen_in_source_at is None:
                row.last_seen_in_source_at = row.first_seen_at

    @staticmethod
    def _seed_imported_universe_status_events(
        db: Session,
        *,
        rows: Iterable[StockUniverse],
        baseline_at: datetime | None,
        source_revision: str | None,
    ) -> int:
        if baseline_at is None:
            return 0
        imported_rows = tuple(rows)
        if not imported_rows:
            return 0

        symbols = tuple(row.symbol for row in imported_rows)
        existing_events = (
            db.query(StockUniverseStatusEvent)
            .filter(
                StockUniverseStatusEvent.symbol.in_(symbols),
                StockUniverseStatusEvent.event_type == UNIVERSE_EVENT_STATUS_CHANGED,
            )
            .order_by(
                StockUniverseStatusEvent.symbol.asc(),
                StockUniverseStatusEvent.created_at.desc(),
                StockUniverseStatusEvent.id.desc(),
            )
            .all()
        )
        events_by_symbol: dict[str, list[StockUniverseStatusEvent]] = {}
        for event in existing_events:
            events_by_symbol.setdefault(event.symbol, []).append(event)

        events: list[StockUniverseStatusEvent] = []
        for row in imported_rows:
            new_status = row.status or (
                UNIVERSE_STATUS_ACTIVE if row.is_active else None
            )
            if not new_status:
                continue
            baseline_utc = _as_utc_datetime(baseline_at)
            event_at = max(
                _as_utc_datetime(row.first_seen_at or baseline_at),
                baseline_utc,
            )
            cutoff_at = baseline_utc + timedelta(days=1)
            timezone_name = row.timezone
            if not timezone_name and row.market:
                try:
                    timezone_name = market_registry.profile(row.market).timezone_name
                except ValueError:
                    timezone_name = None
            if timezone_name:
                try:
                    market_timezone = ZoneInfo(timezone_name)
                    cutoff_at = datetime.combine(
                        baseline_utc.astimezone(market_timezone).date()
                        + timedelta(days=1),
                        time.min,
                        tzinfo=market_timezone,
                    ).astimezone(timezone.utc)
                except ZoneInfoNotFoundError:
                    pass

            latest_before_cutoff = None
            for event in events_by_symbol.get(row.symbol, []):
                if event.created_at is None:
                    continue
                if _as_utc_datetime(event.created_at) < cutoff_at:
                    latest_before_cutoff = event
                    break
            if (
                latest_before_cutoff is not None
                and latest_before_cutoff.new_status == new_status
            ):
                continue

            old_status = (
                latest_before_cutoff.new_status
                if latest_before_cutoff is not None
                else None
            )
            if (
                latest_before_cutoff is not None
                and latest_before_cutoff.created_at is not None
            ):
                latest_at = _as_utc_datetime(latest_before_cutoff.created_at)
                if event_at <= latest_at:
                    next_event_at = latest_at + timedelta(microseconds=1)
                    event_at = (
                        next_event_at if next_event_at < cutoff_at else latest_at
                    )
            events.append(
                StockUniverseStatusEvent(
                    symbol=row.symbol,
                    event_type=UNIVERSE_EVENT_STATUS_CHANGED,
                    old_status=old_status,
                    new_status=new_status,
                    trigger_source="weekly_reference_import",
                    reason="Seeded lifecycle status from weekly reference bundle",
                    payload_json=json.dumps(
                        {
                            "market": row.market,
                            "source_revision": source_revision,
                        },
                        sort_keys=True,
                    ),
                    created_at=event_at,
                )
            )
        if events:
            db.add_all(events)
        return len(events)

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
        force_publish: bool = False,
    ) -> Dict[str, Any]:
        """Publish a pre-built market snapshot using the standard run/pointer tables.

        ``force_publish`` lets the caller publish even when the coverage gate
        would otherwise mark the run as ``publish_blocked``. Used by the weekly
        builder when a deadline-driven partial run still produces a usable
        bundle (skipped symbols inherit data from the prior bundle). The gate's
        warnings are still recorded on the run so consumers can detect the
        degraded state.
        """
        normalized_market = str(market or "").strip().upper()
        parity_stats = {
            "market": normalized_market,
            "missing_active_symbols": [],
        }
        coverage_ok, coverage_warnings, coverage_thresholds = self._coverage_gate(
            coverage_stats,
            market=normalized_market,
        )
        all_warnings = list(warnings or [])
        all_warnings.extend(coverage_warnings)
        force_override = bool(force_publish) and not coverage_ok and publish
        if force_override:
            all_warnings.append(
                "force_publish=True: publishing despite coverage gate failure "
                f"(active_coverage={coverage_thresholds.get('active_coverage', 0.0):.2%}, "
                f"min={coverage_thresholds.get('min_active_coverage', 0.0):.2%})"
            )

        run = ProviderSnapshotRun(
            snapshot_key=snapshot_key,
            run_mode=run_mode,
            status="building",
            source_revision=source_revision,
        )
        db.add(run)
        db.flush()

        # Defense-in-depth for the (run_id, symbol) unique constraint: collapse any
        # rows that share a symbol before insert, so a collision can never abort the
        # whole market build. build_market_snapshot_row now trusts the already-unique
        # StockUniverse.symbol, so the primary collision source (re-canonicalizing a
        # drifted TW .TWO/.TW pair) is gone; this guards only the residual case where
        # light normalization maps two stored symbols together. Mirrors the US path,
        # which already dedups via a symbol-keyed dict. Last write wins; logged.
        deduped_by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            symbol = row["symbol"]
            if symbol in deduped_by_symbol:
                logger.warning(
                    "Collapsing duplicate snapshot row for %s in %s run (canonical "
                    "symbol collision); keeping last occurrence.",
                    symbol,
                    normalized_market,
                )
            deduped_by_symbol[symbol] = row
        materialized_rows = list(deduped_by_symbol.values())
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

        # ``coverage_stats`` is computed by the caller before this method dedups, so
        # clamp the persisted counts to the actual number of rows written — the run
        # metadata must never claim more symbols than exist in provider_snapshot_row.
        persisted_count = len(materialized_rows)

        def _count(key: str) -> int:
            return min(int(coverage_stats.get(key, persisted_count) or 0), persisted_count)

        run.symbols_total = _count("snapshot_symbols")
        run.symbols_published = _count("covered_active_symbols")
        run.coverage_stats_json = json.dumps(coverage_stats, sort_keys=True)
        run.parity_stats_json = json.dumps(parity_stats, sort_keys=True)
        run.warnings_json = json.dumps(all_warnings, sort_keys=True) if all_warnings else None
        run.status = "preview_ready" if not publish else "published"
        published = False
        if publish and (coverage_ok or force_override):
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
            "coverage_thresholds": coverage_thresholds,
            "force_published": force_override,
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

            # Finviz is the primary source for market_cap; fall back to Yahoo
            # only when Finviz's snapshot omitted it. Treat NaN/inf from
            # .info (common for delisted / illiquid tickers) as missing so
            # downstream FX normalization can't raise on ``int(nan * rate)``.
            info_market_cap = _finite_or_none(info.get("marketCap"))
            info_shares = _finite_or_none(info.get("sharesOutstanding"))
            if info_market_cap is None or info_shares is None:
                fast_market_cap, fast_shares, _ = (
                    self.bulk_fetcher._read_fast_info_market_state(ticker)
                )
                if info_market_cap is None:
                    info_market_cap = fast_market_cap
                if info_shares is None:
                    info_shares = fast_shares
            if info_market_cap is not None:
                yahoo_payload["market_cap"] = info_market_cap
            if info_shares is not None:
                yahoo_payload["shares_outstanding"] = info_shares

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
        normalized_market = (
            str(market or self.market_for_snapshot_key(snapshot_key)).strip().upper()
        )
        active_rows_query = db.query(StockUniverse).filter(StockUniverse.active_filter())
        if normalized_market in WEEKLY_REFERENCE_MARKETS:
            active_rows_query = active_rows_query.filter(
                StockUniverse.market == normalized_market
            )
        active_rows = active_rows_query.all()
        active_symbols = {row.symbol for row in active_rows}
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
        coverage_ok, coverage_warnings, coverage_thresholds = self._coverage_gate(
            coverage_stats,
            market=normalized_market,
        )

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
            "coverage_thresholds": coverage_thresholds,
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

    def _validate_weekly_reference_manifest_freshness(
        self,
        manifest: dict[str, Any],
    ) -> tuple[bool, str | None]:
        as_of_raw = str(manifest.get("as_of_date") or "")
        if not as_of_raw:
            raise ValueError("Weekly reference manifest is missing as_of_date")
        as_of_date = date.fromisoformat(as_of_raw)
        max_age_days = max(int(settings.github_weekly_reference_max_age_days or 0), 0)
        if max_age_days and (date.today() - as_of_date).days > max_age_days:
            return (
                True,
                f"Weekly reference bundle is older than the configured max age of {max_age_days} day(s)",
            )
        return False, None

    def sync_weekly_reference_from_github(
        self,
        db: Session,
        *,
        market: str,
        hydrate_cache: bool = True,
        hydrate_mode: Literal["static", "full"] = "static",
        allow_stale: bool = False,
        github_sync_service: GitHubReleaseSyncService | None = None,
    ) -> Dict[str, Any]:
        normalized_market = str(market or "").strip().upper()
        if normalized_market not in WEEKLY_REFERENCE_MARKETS:
            return {
                "status": "unsupported_market",
                "market": normalized_market,
                "source": "github",
            }

        snapshot_key = self.snapshot_key_for_market(normalized_market)
        current_run = self.get_published_run(db, snapshot_key=snapshot_key)
        current_revision = current_run.source_revision if current_run is not None else None
        sync_service = github_sync_service or GitHubReleaseSyncService(
            api_base=settings.github_data_api_base
        )
        download_dir = Path(
            tempfile.mkdtemp(prefix=f"weekly-reference-{normalized_market.lower()}-")
        )
        sync_result = fetch_weekly_reference_bundle(
            sync_service=sync_service,
            market=normalized_market,
            current_revision=current_revision,
            output_dir=download_dir,
            stale_validator=self._validate_weekly_reference_manifest_freshness,
            allow_stale=allow_stale,
        )
        sync_result["source"] = "github"
        sync_result["market"] = normalized_market

        bundle_path = sync_result.get("bundle_path")
        if sync_result.get("status") != "success":
            shutil.rmtree(download_dir, ignore_errors=True)
            return sync_result

        try:
            import_stats = self.import_weekly_reference_bundle(
                db,
                input_path=Path(str(bundle_path)),
                hydrate_cache=hydrate_cache,
                hydrate_mode=hydrate_mode,
            )
        finally:
            if bundle_path:
                Path(str(bundle_path)).unlink(missing_ok=True)
            shutil.rmtree(download_dir, ignore_errors=True)

        return {
            **sync_result,
            "import": import_stats,
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
        hydrate_cache: bool = True,
        hydrate_mode: Literal["static", "full"] = "static",
    ) -> Dict[str, Any]:
        """Import a weekly reference bundle into the local database."""
        if hydrate_mode not in {"static", "full"}:
            raise ValueError(f"Unsupported hydrate_mode {hydrate_mode!r}")
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
        imported_payloads: list[dict[str, Any]] = []
        lifecycle_event_baseline_at = self._weekly_reference_lifecycle_baseline_at(
            payload,
            snapshot,
            market=bundle_market,
        )
        lifecycle_event_source_revision = snapshot.get("source_revision")

        try:
            self._replace_snapshot_key_runs(db, snapshot_key=snapshot_key)
            if legacy_global_bundle:
                imported_universe_count = self._replace_all_universe_rows(
                    db,
                    rows=universe_rows,
                    lifecycle_event_baseline_at=lifecycle_event_baseline_at,
                    lifecycle_event_source_revision=lifecycle_event_source_revision,
                )
            else:
                imported_universe_count = self._replace_market_universe_rows(
                    db,
                    market=bundle_market,
                    rows=universe_rows,
                    lifecycle_event_baseline_at=lifecycle_event_baseline_at,
                    lifecycle_event_source_revision=lifecycle_event_source_revision,
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

            # Build + dedup snapshot rows by canonical symbol (see
            # _deserialize_snapshot_rows): a bundle's .TWO/.TW phantom pair
            # re-canonicalizes to one symbol and would otherwise violate
            # uq_provider_snapshot_row_run_symbol. Row and payload stay in lockstep.
            deduped = self._deserialize_snapshot_rows(
                snapshot_rows, run_id=run.id, bundle_market=bundle_market
            )
            rows = [snapshot_row for snapshot_row, _ in deduped.values()]
            imported_payloads.extend(payload for _, payload in deduped.values())
            if rows:
                db.bulk_save_objects(rows)
            # Dedup may have reduced the snapshot-row count below what the bundle's
            # symbols_total/published claim; clamp so run metadata never exceeds the
            # rows actually persisted. (No-op for non-colliding bundles.)
            run.symbols_total = min(run.symbols_total, len(rows))
            run.symbols_published = min(run.symbols_published, len(rows))

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

        hydrated_symbols = 0
        failed_hydration_symbols: list[str] = []
        if hydrate_cache and imported_payloads:
            existing_payloads = (
                self.fundamentals_cache.get_many(
                    [payload["symbol"] for payload in imported_payloads]
                )
                if hydrate_mode == "full"
                else {}
            )
            for normalized_payload in imported_payloads:
                symbol = normalized_payload["symbol"]
                payload_to_store = dict(normalized_payload)
                if hydrate_mode == "full":
                    payload_to_store = self.fundamentals_cache._merge_fundamentals(
                        payload_to_store,
                        existing_payloads.get(symbol) or {},
                    )
                persisted = self.fundamentals_cache.store(
                    symbol,
                    payload_to_store,
                    data_source="bundle_import",
                    market=payload_to_store.get("market"),
                )
                if persisted:
                    hydrated_symbols += 1
                else:
                    failed_hydration_symbols.append(symbol)

        if failed_hydration_symbols:
            preview = ", ".join(failed_hydration_symbols[:10])
            if len(failed_hydration_symbols) > 10:
                preview += ", ..."
            raise RuntimeError(
                "Failed to persist imported fundamentals for "
                f"{len(failed_hydration_symbols)} symbol(s): {preview}"
            )

        return {
            "run_id": run.id,
            "source_revision": run.source_revision,
            "rows": len(rows),
            "universe_rows": imported_universe_count,
            "as_of_date": payload.get("as_of_date"),
            "market": "MULTI" if legacy_global_bundle else bundle_market,
            "hydrate_cache": hydrate_cache,
            "hydrate_mode": hydrate_mode,
            "hydrated_symbols": hydrated_symbols,
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

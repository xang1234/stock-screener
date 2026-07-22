"""
Stock Universe Service for managing scannable stock lists.

Fetches stocks from finviz and manages the stock_universe database table.
"""
import logging
import csv
import hashlib
import io
import json
import os
from time import sleep
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
import pandas as pd
from typing import Any, Dict, Iterable, List, Mapping, Optional
from finvizfinance.util import web_scrap
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_
from datetime import datetime, timedelta, timezone

from ..models.stock_universe import (
    StockUniverse,
    StockUniverseIndexMembership,
    StockUniverseReconciliationRun,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
    UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
    UNIVERSE_STATUS_INACTIVE_NO_DATA,
)
from ..models.stock import StockIndustry
from ..models.ticker_validation import TickerValidationLog
from .universe_classification import prefer_meaningful
from ..schemas.universe import IndexName
from ..config import settings
from ..domain.markets.catalog import get_market_catalog
from ..domain.markets.mic_aliases import mic_alias_registry
from ..domain.universe.ingestion import (
    CanonicalUniverseIngestionResult,
    CanonicalUniverseRow,
    RejectedUniverseRow,
    UniverseCoverageRejection,
    UniverseIngestionContext,
    UniverseIngestionSideEffects,
    UniverseReconciliationPolicy,
)
from .au_universe_ingestion_adapter import au_universe_ingestion_adapter
from .ca_universe_ingestion_adapter import ca_universe_ingestion_adapter
from .cn_universe_ingestion_adapter import cn_universe_ingestion_adapter
from .de_universe_ingestion_adapter import de_universe_ingestion_adapter
from .finviz_universe_ingestion_adapter import finviz_universe_ingestion_adapter
from .hk_universe_ingestion_adapter import hk_universe_ingestion_adapter
from .jp_universe_ingestion_adapter import jp_universe_ingestion_adapter
from .kr_universe_ingestion_adapter import kr_universe_ingestion_adapter
from .my_universe_ingestion_adapter import my_universe_ingestion_adapter
from .security_master_service import security_master_resolver
from .sg_universe_ingestion_adapter import sg_universe_ingestion_adapter
from .tw_universe_ingestion_adapter import tw_universe_ingestion_adapter
from .universe_ingestion_pipeline import (
    FlatUniverseCanonicalizerAdapter,
    UniverseBeforeReconciliationContext,
    UniverseIngestionPipeline,
    UniversePersistence,
)

logger = logging.getLogger(__name__)

_MARKET_CATALOG = get_market_catalog()
MARKET_EXCHANGE_FALLBACKS: dict[str, tuple[str, ...]] = {
    code: _MARKET_CATALOG.get(code).exchanges
    for code in _MARKET_CATALOG.supported_market_codes()
}
KR_ACTIVE_UNIVERSE_MIN_COUNT = 2526
CN_ACTIVE_UNIVERSE_MIN_COUNT = 5217
# Provisional: estimated from public TMX issuer counts (~3,200 raw, ~2,300-2,600
# after ETF/fund/debt/derivative exclusions). Re-baseline against the first
# successful TMX snapshot before treating as authoritative.
CA_ACTIVE_UNIVERSE_MIN_COUNT = 2300
FINVIZ_EXCHANGE_FILTER_CODES = {
    "AMEX": "amex",
    "NASDAQ": "nasd",
    "NYSE": "nyse",
}
FINVIZ_SCREENER_URL = "https://finviz.com/screener.ashx"
FINVIZ_OVERVIEW_PAGE = 111
FINVIZ_SCREENER_PAGE_SIZE = 20


class StockUniverseService:
    """Service for managing stock universe (NYSE/NASDAQ stocks from finviz)"""

    def __init__(self):
        """Initialize stock universe service."""
        self._security_master = security_master_resolver
        self._au_ingestion = au_universe_ingestion_adapter
        self._ca_ingestion = ca_universe_ingestion_adapter
        self._cn_ingestion = cn_universe_ingestion_adapter
        self._de_ingestion = de_universe_ingestion_adapter
        self._finviz_ingestion = finviz_universe_ingestion_adapter
        self._hk_ingestion = hk_universe_ingestion_adapter
        self._jp_ingestion = jp_universe_ingestion_adapter
        self._kr_ingestion = kr_universe_ingestion_adapter
        self._my_ingestion = my_universe_ingestion_adapter
        self._sg_ingestion = sg_universe_ingestion_adapter
        self._tw_ingestion = tw_universe_ingestion_adapter
        self._universe_ingestion_pipeline = UniverseIngestionPipeline(
            canonicalizers={
                "US": self._finviz_ingestion,
                "AU": self._au_ingestion,
                "HK": FlatUniverseCanonicalizerAdapter(self._hk_ingestion),
                "JP": FlatUniverseCanonicalizerAdapter(self._jp_ingestion),
                "KR": FlatUniverseCanonicalizerAdapter(self._kr_ingestion),
                "TW": FlatUniverseCanonicalizerAdapter(self._tw_ingestion),
                "CA": FlatUniverseCanonicalizerAdapter(self._ca_ingestion),
                "DE": FlatUniverseCanonicalizerAdapter(self._de_ingestion),
                "CN": FlatUniverseCanonicalizerAdapter(self._cn_ingestion),
                "SG": self._sg_ingestion,
                "MY": FlatUniverseCanonicalizerAdapter(self._my_ingestion),
            },
            persistence=UniversePersistence.for_stock_universe_service(self),
            before_reconciliation_hooks={
                "CN": self._upsert_cn_stock_industry_from_pipeline_context,
                "IN": self._deactivate_india_coverage_rejections,
            },
        )
        self._bulk_fetcher = None

    @staticmethod
    def _utc_iso(value: datetime | None) -> str | None:
        """Serialize datetimes in UTC ISO-8601 format for API payloads."""
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return f"{value.isoformat()}Z"

    @staticmethod
    def _snapshot_age_seconds(*, now: datetime, value: datetime | None) -> int | None:
        """Return non-negative age in seconds from now to value."""
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return max(int((now - value).total_seconds()), 0)

    @staticmethod
    def _normalize_status_fields(status: str | None, is_active: bool | None) -> str:
        """Return lifecycle status from raw columns without full ORM object hydration."""
        raw_status = (status or "").strip()
        active_flag = bool(is_active)
        if raw_status == UNIVERSE_STATUS_ACTIVE and not active_flag:
            return UNIVERSE_STATUS_INACTIVE_MANUAL
        if raw_status and raw_status != UNIVERSE_STATUS_ACTIVE and active_flag:
            return UNIVERSE_STATUS_ACTIVE
        if raw_status:
            return raw_status
        return UNIVERSE_STATUS_ACTIVE if active_flag else UNIVERSE_STATUS_INACTIVE_MANUAL

    def _resolved_identity(self, stock_data: Dict[str, Any]):
        return self._security_master.resolve_identity(
            symbol=stock_data.get("symbol", ""),
            market=stock_data.get("market"),
            exchange=stock_data.get("exchange"),
        )

    @staticmethod
    def _auto_snapshot_id(prefix: str) -> str:
        """Generate collision-resistant snapshot IDs for implicit ingestion runs."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}:{timestamp}:{uuid4().hex[:8]}"

    @staticmethod
    def _normalize_status(record: StockUniverse) -> str:
        """Return a lifecycle status even for pre-migration rows."""
        raw_status = (record.status or "").strip()
        if raw_status == UNIVERSE_STATUS_ACTIVE and record.is_active is False:
            return UNIVERSE_STATUS_INACTIVE_MANUAL
        if raw_status and raw_status != UNIVERSE_STATUS_ACTIVE and record.is_active is True:
            return UNIVERSE_STATUS_ACTIVE
        if raw_status:
            return raw_status
        return UNIVERSE_STATUS_ACTIVE if record.is_active else UNIVERSE_STATUS_INACTIVE_MANUAL

    def _add_status_event(
        self,
        db: Session,
        *,
        symbol: str,
        old_status: Optional[str],
        new_status: str,
        trigger_source: str,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        db.add(
            self._build_status_event_record(
                symbol=symbol,
                old_status=old_status,
                new_status=new_status,
                trigger_source=trigger_source,
                reason=reason,
                payload=payload,
            )
        )

    @staticmethod
    def _build_status_event_record(
        *,
        symbol: str,
        old_status: Optional[str],
        new_status: str,
        trigger_source: str,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> StockUniverseStatusEvent:
        return StockUniverseStatusEvent(
            symbol=symbol,
            event_type=UNIVERSE_EVENT_STATUS_CHANGED,
            old_status=old_status,
            new_status=new_status,
            trigger_source=trigger_source,
            reason=reason,
            payload_json=json.dumps(payload, sort_keys=True) if payload else None,
        )

    @staticmethod
    def _build_metadata_event_record(
        *,
        symbol: str,
        event_type: str,
        trigger_source: str,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> StockUniverseStatusEvent:
        if event_type == UNIVERSE_EVENT_STATUS_CHANGED:
            raise ValueError("Use _build_status_event_record for lifecycle events")
        return StockUniverseStatusEvent(
            symbol=symbol,
            event_type=event_type,
            old_status=None,
            new_status=None,
            trigger_source=trigger_source,
            reason=reason,
            payload_json=json.dumps(payload, sort_keys=True) if payload else None,
        )

    @staticmethod
    def _bulk_insert_records(db: Session, objects: list[Any]) -> None:
        # bulk_save_objects skips Python-side Column(default=...) values, so
        # callers must populate any required non-server defaults up front.
        if objects:
            db.bulk_save_objects(objects)

    @staticmethod
    def _parse_optional_float(value: Any) -> float | None:
        raw = str(value or "").strip().replace(",", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _get_bulk_fetcher(self):
        if self._bulk_fetcher is None:
            from .bulk_data_fetcher import BulkDataFetcher

            self._bulk_fetcher = BulkDataFetcher()
        return self._bulk_fetcher

    @staticmethod
    def _is_india_bse_exchange(exchange: str | None) -> bool:
        return str(exchange or "").strip().upper() in {"BSE", "XBOM"}

    @staticmethod
    def _has_verified_yfinance_price_coverage(result: Mapping[str, Any] | None) -> bool:
        if not isinstance(result, Mapping):
            return False
        if bool(result.get("has_error")):
            return False
        price_data = result.get("price_data")
        if price_data is None:
            return False
        try:
            return len(price_data) > 0
        except TypeError:
            return False

    def _india_bse_symbols_with_unresolved_yfinance_failures(
        self,
        db: Session,
        *,
        symbols: Iterable[str],
    ) -> set[str]:
        normalized_symbols = sorted(
            {
                str(symbol or "").strip().upper()
                for symbol in symbols
                if str(symbol or "").strip()
            }
        )
        if not normalized_symbols:
            return set()

        cutoff = datetime.utcnow() - timedelta(days=settings.india_bse_validation_days_back)
        rows = (
            db.query(TickerValidationLog.symbol)
            .filter(
                TickerValidationLog.symbol.in_(normalized_symbols),
                TickerValidationLog.is_resolved.is_(False),
                TickerValidationLog.detected_at >= cutoff,
                TickerValidationLog.consecutive_failures
                >= settings.india_bse_validation_failures_threshold,
                TickerValidationLog.data_source.in_(("yfinance", "both")),
                TickerValidationLog.error_type.in_(
                    ("no_data", "empty_info", "invalid_response", "delisted")
                ),
            )
            .distinct()
            .all()
        )
        return {str(row[0]).strip().upper() for row in rows if row and row[0]}

    def _filter_india_bse_rows_for_downstream_support(
        self,
        db: Session,
        rows: Iterable[SimpleNamespace],
    ) -> tuple[list[SimpleNamespace], list[SimpleNamespace]]:
        normalized_rows = list(rows)
        if not settings.india_bse_coverage_gate_enabled:
            return normalized_rows, []

        bse_rows = [row for row in normalized_rows if self._is_india_bse_exchange(row.exchange)]
        if not bse_rows:
            return normalized_rows, []

        blocked_symbols = self._india_bse_symbols_with_unresolved_yfinance_failures(
            db,
            symbols=(row.symbol for row in bse_rows),
        )
        symbols_to_verify = [row.symbol for row in bse_rows if row.symbol not in blocked_symbols]
        verification_results: dict[str, dict[str, Any]] = {}
        if symbols_to_verify:
            verification_results = self._get_bulk_fetcher().fetch_prices_in_batches(
                symbols_to_verify,
                period=settings.india_bse_price_verification_period,
                market="IN",
            )
            verified_count = sum(
                1
                for symbol in symbols_to_verify
                if self._has_verified_yfinance_price_coverage(verification_results.get(symbol))
            )
            if (
                verified_count == 0
                and len(symbols_to_verify) >= settings.india_bse_gate_global_failure_min_symbols
            ):
                raise ValueError(
                    "India BSE coverage gate could not verify any BSE-only symbols via yfinance"
                )

        accepted_rows: list[SimpleNamespace] = []
        rejected_rows: list[SimpleNamespace] = []
        for row in normalized_rows:
            if not self._is_india_bse_exchange(row.exchange):
                accepted_rows.append(row)
                continue
            if row.symbol in blocked_symbols:
                rejected_rows.append(
                    SimpleNamespace(
                        source_row_number=row.source_row_number,
                        source_symbol=row.source_symbol,
                        symbol=row.symbol,
                        reason="unresolved_yfinance_validation_failures",
                    )
                )
                continue
            if not self._has_verified_yfinance_price_coverage(verification_results.get(row.symbol)):
                rejected_rows.append(
                    SimpleNamespace(
                        source_row_number=row.source_row_number,
                        source_symbol=row.source_symbol,
                        symbol=row.symbol,
                        reason="missing_yfinance_price_coverage",
                    )
                )
                continue
            accepted_rows.append(row)

        return accepted_rows, rejected_rows

    def _canonicalize_in_snapshot_rows(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[list[SimpleNamespace], list[SimpleNamespace]]:
        normalized_source_name = str(source_name or "").strip().lower().replace("-", "_")
        if not normalized_source_name:
            raise ValueError("source_name must be provided")

        canonical_rows: list[SimpleNamespace] = []
        rejected_rows: list[SimpleNamespace] = []
        seen_symbols: set[str] = set()

        for row_number, raw_row in enumerate(rows, start=1):
            row = dict(raw_row or {})
            source_symbol = (
                str(row.get("symbol") or row.get("local_code") or row.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol:
                rejected_rows.append(
                    SimpleNamespace(
                        source_row_number=row_number,
                        source_symbol="",
                        reason="missing symbol",
                    )
                )
                continue

            exchange = str(row.get("exchange") or "").strip().upper() or None
            name = str(row.get("name") or "").strip()
            isin = str(row.get("isin") or "").strip().upper()
            if not name:
                rejected_rows.append(
                    SimpleNamespace(
                        source_row_number=row_number,
                        source_symbol=source_symbol,
                        reason="missing name",
                    )
                )
                continue

            try:
                identity = self._security_master.resolve_identity(
                    symbol=source_symbol,
                    market="IN",
                    exchange=exchange,
                )
            except Exception as exc:
                rejected_rows.append(
                    SimpleNamespace(
                        source_row_number=row_number,
                        source_symbol=source_symbol,
                        reason=str(exc),
                    )
                )
                continue

            if identity.canonical_symbol in seen_symbols:
                continue
            seen_symbols.add(identity.canonical_symbol)

            row_payload = {
                "symbol": identity.canonical_symbol,
                "name": name,
                "market": identity.market,
                "exchange": identity.exchange,
                "currency": identity.currency,
                "timezone": identity.timezone,
                "local_code": identity.local_code,
                "sector": str(row.get("sector") or "").strip(),
                "industry": str(row.get("industry") or "").strip(),
                "market_cap": self._parse_optional_float(row.get("market_cap")),
            }
            lineage_payload = {
                "source_name": normalized_source_name,
                "source_symbol": source_symbol,
                "source_row_number": row_number,
                "snapshot_id": snapshot_id,
                "snapshot_as_of": snapshot_as_of,
                "source_metadata": dict(source_metadata or {}),
                "raw_row": row,
                "isin": isin,
            }
            canonical_rows.append(
                SimpleNamespace(
                    **row_payload,
                    source_name=normalized_source_name,
                    source_symbol=source_symbol,
                    source_row_number=row_number,
                    snapshot_id=snapshot_id,
                    snapshot_as_of=snapshot_as_of,
                    source_metadata={
                        **dict(source_metadata or {}),
                        **({"isin": isin} if isin else {}),
                        **(
                            {"security_id": str(row.get("security_id") or "").strip().upper()}
                            if str(row.get("security_id") or "").strip()
                            else {}
                        ),
                    },
                    lineage_hash=self._sha256_text(self._stable_json(lineage_payload)),
                    row_hash=self._sha256_text(self._stable_json(row_payload)),
                )
            )

        return canonical_rows, rejected_rows

    def _apply_status_transition(
        self,
        db: Session,
        record: StockUniverse,
        *,
        new_status: str,
        trigger_source: str,
        reason: str,
        now: Optional[datetime] = None,
        payload: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        clear_failures: bool = False,
        seen_in_source: bool = False,
    ) -> bool:
        """Apply a lifecycle transition and emit an audit event when status changes."""
        now = now or datetime.utcnow()
        old_status = self._normalize_status(record)

        record.status = new_status
        record.is_active = new_status == UNIVERSE_STATUS_ACTIVE
        record.status_reason = reason
        record.updated_at = now

        if source:
            record.source = source

        if record.first_seen_at is None:
            record.first_seen_at = now

        if new_status == UNIVERSE_STATUS_ACTIVE:
            record.deactivated_at = None
            if seen_in_source:
                record.last_seen_in_source_at = now
            if clear_failures:
                record.consecutive_fetch_failures = 0
                record.last_fetch_failure_at = None
        else:
            if record.deactivated_at is None or old_status != new_status:
                record.deactivated_at = now

        changed = old_status != new_status
        if changed:
            self._add_status_event(
                db,
                symbol=record.symbol,
                old_status=old_status,
                new_status=new_status,
                trigger_source=trigger_source,
                reason=reason,
                payload=payload,
            )
        return changed

    def fetch_from_finviz(self, exchange_filter: Optional[str] = None) -> List[Dict]:
        """
        Fetch all US stocks from the Finviz screener.

        Args:
            exchange_filter: Optional filter: 'nyse', 'nasdaq', 'amex', or None for all

        Returns:
            List of stock data dicts with symbol, name, sector, industry, market_cap, exchange
        """
        try:
            logger.info(f"Fetching stocks from finviz (exchange_filter={exchange_filter})")

            # If no exchange filter, we'll fetch from all major exchanges
            # We'll do this by fetching each exchange separately and combining
            if not exchange_filter:
                all_stocks = []
                for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
                    try:
                        logger.info(f"Fetching stocks from {exchange}...")

                        stocks = self._fetch_finviz_exchange_rows(exchange)
                        if stocks:
                            all_stocks.extend(stocks)
                        logger.info(f"Fetched {len(stocks)} stocks from {exchange}")
                    except Exception as e:
                        logger.warning(f"Error fetching from {exchange}: {e}")
                        continue

                logger.info(f"Successfully fetched {len(all_stocks)} total stocks from all exchanges")
                return all_stocks
            else:
                # Fetch from specific exchange
                exchange_name = exchange_filter.upper()
                if exchange_name not in ['NYSE', 'NASDAQ', 'AMEX']:
                    logger.warning(f"Invalid exchange filter: {exchange_filter}")
                    return []

                stocks = self._fetch_finviz_exchange_rows(exchange_name)
                if not stocks:
                    logger.warning(f"No data returned from finviz for exchange {exchange_filter}")
                    return []

                logger.info(f"Successfully fetched {len(stocks)} stocks from {exchange_name}")
                return stocks

        except Exception as e:
            logger.error(f"Error fetching stocks from finviz: {e}", exc_info=True)
            return []

    def _fetch_finviz_exchange_rows(self, exchange: str) -> List[Dict[str, Any]]:
        params: dict[str, Any] = {
            "v": FINVIZ_OVERVIEW_PAGE,
            "f": f"exch_{FINVIZ_EXCHANGE_FILTER_CODES[exchange]}",
            "o": "ticker",
        }
        stocks: list[dict[str, Any]] = []

        soup = web_scrap(FINVIZ_SCREENER_URL, params)
        page_count = self._finviz_page_count(soup)
        stocks.extend(self._parse_finviz_screener_soup(soup, exchange))

        for page_index in range(1, page_count):
            sleep(1)
            params["r"] = page_index * FINVIZ_SCREENER_PAGE_SIZE + 1
            soup = web_scrap(FINVIZ_SCREENER_URL, params)
            stocks.extend(self._parse_finviz_screener_soup(soup, exchange))

        return stocks

    @staticmethod
    def _finviz_page_count(soup: Any) -> int:
        page_select = soup.find(id="pageSelect")
        if page_select is None:
            return 0
        return len(page_select.find_all("option"))

    def _parse_finviz_screener_soup(self, soup: Any, exchange: str) -> list[dict[str, Any]]:
        table = soup.find("table", class_="screener_table")
        if table is None:
            return []

        rows = table.find_all("tr")
        if not rows:
            return []

        headers = [header.get_text(strip=True) for header in rows[0].find_all("th")][1:]
        parsed_rows: list[dict[str, Any]] = []
        for row in rows[1:]:
            cells = row.find_all("td")[1:]
            if not cells:
                continue
            payload: dict[str, Any] = {}
            for header, cell in zip(headers, cells):
                if header == "Ticker":
                    payload[header] = self._extract_finviz_ticker(cell)
                else:
                    payload[header] = cell.get_text(strip=True)
            parsed_rows.append(payload)

        return self._parse_finviz_dataframe(pd.DataFrame(parsed_rows), exchange)

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

    def _parse_finviz_dataframe(self, df, exchange: str) -> List[Dict]:
        """
        Parse finvizfinance DataFrame into list of stock dicts.

        Args:
            df: pandas DataFrame from finvizfinance
            exchange: Exchange name (NYSE, NASDAQ, AMEX)

        Returns:
            List of stock data dicts
        """
        stocks = []

        for idx, row in df.iterrows():
            try:
                # Extract data from DataFrame row
                ticker = row.get('Ticker', '')
                if not ticker:
                    continue

                # Parse market cap
                market_cap_str = row.get('Market Cap', '')
                market_cap = self._parse_market_cap(market_cap_str)

                stock_data = {
                    'symbol': str(ticker).upper(),
                    'name': str(row.get('Company', '')),
                    'sector': str(row.get('Sector', '')),
                    'industry': str(row.get('Industry', '')),
                    'market_cap': market_cap,
                    'exchange': exchange,
                }

                stocks.append(stock_data)

            except Exception as e:
                logger.warning(f"Error parsing stock row: {e}")
                continue

        return stocks

    def import_from_csv(self, csv_content: str) -> List[Dict]:
        """
        Import stocks from CSV file content.

        CSV format: symbol,name,exchange,sector,industry,market_cap
        Header row is optional. Minimum required: symbol

        Args:
            csv_content: CSV file content as string

        Returns:
            List of stock data dicts
        """
        try:
            logger.info("Importing stocks from CSV")
            stocks = []

            # Parse CSV
            csv_file = io.StringIO(csv_content)
            reader = csv.DictReader(csv_file)

            # Check if header exists, if not, assume first row is data
            fieldnames = reader.fieldnames
            if not fieldnames or 'symbol' not in [f.lower() for f in fieldnames]:
                # No header or wrong header, use default fieldnames
                csv_file.seek(0)
                fieldnames = ['symbol', 'name', 'exchange', 'sector', 'industry', 'market_cap']
                reader = csv.DictReader(csv_file, fieldnames=fieldnames)

            for row in reader:
                try:
                    # Normalize keys to lowercase
                    row_lower = {k.lower(): v for k, v in row.items() if k}

                    symbol = row_lower.get('symbol', '').strip().upper()
                    if not symbol or symbol == 'SYMBOL':  # Skip header if present
                        continue

                    # Parse market cap if provided
                    market_cap_str = row_lower.get('market_cap', '') or row_lower.get('marketcap', '')
                    market_cap = self._parse_market_cap(market_cap_str) if market_cap_str else None

                    stock_data = {
                        'symbol': symbol,
                        'name': row_lower.get('name', '').strip() or row_lower.get('company', '').strip() or '',
                        'exchange': row_lower.get('exchange', 'NASDAQ').strip().upper(),
                        'sector': row_lower.get('sector', '').strip() or '',
                        'industry': row_lower.get('industry', '').strip() or '',
                        'market_cap': market_cap,
                    }

                    stocks.append(stock_data)

                except Exception as e:
                    logger.warning(f"Error parsing CSV row: {e}, row: {row}")
                    continue

            logger.info(f"Successfully imported {len(stocks)} stocks from CSV")
            return stocks

        except Exception as e:
            logger.error(f"Error importing from CSV: {e}", exc_info=True)
            return []

    @staticmethod
    def _stable_json(payload: Any) -> str:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )

    @staticmethod
    def _sha256_text(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _legacy_reconciliation_row_payload(self, row: Any) -> dict[str, Any]:
        payload = {
            "symbol": row.symbol,
            "name": row.name,
            "market": row.market,
            "exchange": row.exchange,
            "currency": row.currency,
            "timezone": row.timezone,
            "local_code": row.local_code,
            "sector": row.sector,
            "industry": row.industry,
            "market_cap": float(row.market_cap) if row.market_cap is not None else None,
        }
        payload["content_hash"] = self._sha256_text(self._stable_json(payload))
        return payload

    def _canonical_reconciliation_row_payload(
        self,
        row: CanonicalUniverseRow,
    ) -> dict[str, Any]:
        payload = {
            "symbol": row.symbol,
            "name": row.name,
            "market": row.market,
            "exchange": row.mic,
            "currency": row.currency,
            "timezone": row.timezone,
            "local_code": row.local_code,
            "sector": row.sector,
            "industry": row.industry,
            "market_cap": float(row.market_cap) if row.market_cap is not None else None,
        }
        payload["content_hash"] = self._sha256_text(self._stable_json(payload))
        return payload

    def _reconciliation_row_payloads(self, rows: list[Any]) -> list[dict[str, Any]]:
        canonical_rows = [
            row for row in rows if isinstance(row, CanonicalUniverseRow)
        ]
        if canonical_rows and len(canonical_rows) != len(rows):
            raise ValueError("Cannot mix canonical and legacy Universe reconciliation rows")
        if canonical_rows:
            return [
                self._canonical_reconciliation_row_payload(row)
                for row in canonical_rows
            ]
        return [self._legacy_reconciliation_row_payload(row) for row in rows]

    @staticmethod
    def _reconciliation_changed_fields(
        current: Mapping[str, Any],
        previous: Mapping[str, Any],
    ) -> list[str]:
        fields = (
            "name",
            "market",
            "exchange",
            "currency",
            "timezone",
            "local_code",
            "sector",
            "industry",
            "market_cap",
        )
        return [field for field in fields if current.get(field) != previous.get(field)]

    def _build_market_reconciliation_artifact(
        self,
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        previous_snapshot_id: str | None,
        current_rows: list[dict[str, Any]],
        previous_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current_by_symbol = {str(row["symbol"]): row for row in current_rows}
        previous_by_symbol = {str(row["symbol"]): row for row in previous_rows}

        current_symbols = set(current_by_symbol.keys())
        previous_symbols = set(previous_by_symbol.keys())

        added_symbols = sorted(current_symbols - previous_symbols)
        removed_symbols = sorted(previous_symbols - current_symbols)
        common_symbols = sorted(current_symbols & previous_symbols)

        changed_rows: list[dict[str, Any]] = []
        unchanged_count = 0
        for symbol in common_symbols:
            current = current_by_symbol[symbol]
            previous = previous_by_symbol[symbol]
            changed_fields = self._reconciliation_changed_fields(current, previous)
            if not changed_fields:
                unchanged_count += 1
                continue
            changed_rows.append(
                {
                    "symbol": symbol,
                    "changed_fields": changed_fields,
                    "previous_content_hash": previous.get("content_hash"),
                    "current_content_hash": current.get("content_hash"),
                    "previous": previous,
                    "current": current,
                }
            )

        return {
            "artifact_version": "stock-universe-reconciliation-v1",
            "market": market,
            "source_name": source_name,
            "snapshot_id": snapshot_id,
            "previous_snapshot_id": previous_snapshot_id,
            "counts": {
                "total_current": len(current_rows),
                "total_previous": len(previous_rows),
                "added": len(added_symbols),
                "removed": len(removed_symbols),
                "changed": len(changed_rows),
                "unchanged": unchanged_count,
            },
            "added_symbols": added_symbols,
            "removed_symbols": removed_symbols,
            "changed_rows": changed_rows,
            "snapshot_rows": sorted(current_rows, key=lambda row: str(row["symbol"])),
        }

    def _record_market_reconciliation_run(
        self,
        db: Session,
        *,
        market: str,
        source_name: str,
        snapshot_id: str,
        canonical_rows: Iterable[Any],
    ) -> dict[str, Any]:
        canonical_list = list(canonical_rows)
        normalized_source_name = (
            self._reconciliation_source_name(canonical_list[0])
            if canonical_list
            else str(source_name or "").strip().lower().replace("-", "_")
        )
        current_run = (
            db.query(StockUniverseReconciliationRun)
            .filter(
                StockUniverseReconciliationRun.market == market,
                StockUniverseReconciliationRun.source_name == normalized_source_name,
                StockUniverseReconciliationRun.snapshot_id == snapshot_id,
            )
            .one_or_none()
        )

        # Baseline is immutable per (market, snapshot_id) once the run exists.
        previous_snapshot_id: str | None = (
            current_run.previous_snapshot_id
            if current_run is not None
            else None
        )
        previous_run: StockUniverseReconciliationRun | None = None
        if previous_snapshot_id:
            previous_run = (
                db.query(StockUniverseReconciliationRun)
                .filter(
                    StockUniverseReconciliationRun.market == market,
                    StockUniverseReconciliationRun.source_name == normalized_source_name,
                    StockUniverseReconciliationRun.snapshot_id == previous_snapshot_id,
                )
                .one_or_none()
            )
            if previous_run is None:
                logger.warning(
                    "Reconciliation baseline snapshot not found for existing run",
                    extra={
                        "market": market,
                        "snapshot_id": snapshot_id,
                        "previous_snapshot_id": previous_snapshot_id,
                    },
                )
        elif current_run is None:
            previous_run = (
                db.query(StockUniverseReconciliationRun)
                .filter(
                    StockUniverseReconciliationRun.market == market,
                    StockUniverseReconciliationRun.source_name == normalized_source_name,
                    StockUniverseReconciliationRun.snapshot_id != snapshot_id,
                )
                .order_by(
                    StockUniverseReconciliationRun.created_at.desc(),
                    StockUniverseReconciliationRun.id.desc(),
                )
                .first()
            )
            previous_snapshot_id = previous_run.snapshot_id if previous_run is not None else None

        previous_rows: list[dict[str, Any]] = []
        if previous_run is not None:
            if previous_run.artifact_json:
                try:
                    parsed = json.loads(previous_run.artifact_json)
                    raw_rows = parsed.get("snapshot_rows") if isinstance(parsed, dict) else []
                    if isinstance(raw_rows, list):
                        previous_rows = [row for row in raw_rows if isinstance(row, dict)]
                except Exception:
                    logger.warning(
                        "Unable to parse prior stock universe reconciliation artifact",
                        exc_info=True,
                        extra={
                            "market": market,
                            "snapshot_id": previous_run.snapshot_id,
                        },
                    )

        current_rows = self._reconciliation_row_payloads(canonical_list)
        artifact = self._build_market_reconciliation_artifact(
            market=market,
            source_name=normalized_source_name,
            snapshot_id=snapshot_id,
            previous_snapshot_id=previous_snapshot_id,
            current_rows=current_rows,
            previous_rows=previous_rows,
        )
        artifact_json = self._stable_json(artifact)
        artifact_hash = self._sha256_text(artifact_json)
        counts = artifact["counts"]

        run = current_run
        if run is None:
            run = StockUniverseReconciliationRun(
                market=market,
                source_name=normalized_source_name,
                snapshot_id=snapshot_id,
                previous_snapshot_id=previous_snapshot_id,
                total_current=int(counts["total_current"]),
                total_previous=int(counts["total_previous"]),
                added_count=int(counts["added"]),
                removed_count=int(counts["removed"]),
                changed_count=int(counts["changed"]),
                unchanged_count=int(counts["unchanged"]),
                artifact_hash=artifact_hash,
                artifact_json=artifact_json,
            )
            db.add(run)
        else:
            run.source_name = normalized_source_name
            run.total_current = int(counts["total_current"])
            run.total_previous = int(counts["total_previous"])
            run.added_count = int(counts["added"])
            run.removed_count = int(counts["removed"])
            run.changed_count = int(counts["changed"])
            run.unchanged_count = int(counts["unchanged"])
            run.artifact_hash = artifact_hash
            run.artifact_json = artifact_json

        db.flush()

        details_limit = 25
        changed_rows = artifact.get("changed_rows", [])
        changed_symbols = [str(row.get("symbol", "")) for row in changed_rows if row.get("symbol")]
        added_symbols = artifact.get("added_symbols", [])
        removed_symbols = artifact.get("removed_symbols", [])
        return {
            "run_id": run.id,
            "market": market,
            "source_name": normalized_source_name,
            "snapshot_id": snapshot_id,
            "previous_snapshot_id": previous_snapshot_id,
            "artifact_hash": artifact_hash,
            "counts": {
                "total_current": int(counts["total_current"]),
                "total_previous": int(counts["total_previous"]),
                "added": int(counts["added"]),
                "removed": int(counts["removed"]),
                "changed": int(counts["changed"]),
                "unchanged": int(counts["unchanged"]),
            },
            "added_symbols": added_symbols[:details_limit],
            "removed_symbols": removed_symbols[:details_limit],
            "changed_symbols": changed_symbols[:details_limit],
            "added_symbols_truncated": len(added_symbols) > details_limit,
            "removed_symbols_truncated": len(removed_symbols) > details_limit,
            "changed_symbols_truncated": len(changed_symbols) > details_limit,
        }

    @staticmethod
    def _reconciliation_source_name(row: Any) -> str:
        if isinstance(row, CanonicalUniverseRow):
            return row.provenance.source_name
        return str(row.source_name)

    @staticmethod
    def _min_count_threshold_for_market(market: str) -> int:
        default_min = StockUniverseService._safety_int_setting(
            field_name="asia_reconciliation_min_count_default",
            env_name="ASIA_RECONCILIATION_MIN_COUNT_DEFAULT",
            default=0,
        )
        normalized_market = (market or "").strip().upper()
        if normalized_market == "HK":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_hk",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_HK",
                default=default_min,
            )
        if normalized_market == "JP":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_jp",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_JP",
                default=default_min,
            )
        if normalized_market == "KR":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_kr",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_KR",
                default=max(default_min, KR_ACTIVE_UNIVERSE_MIN_COUNT),
            )
        if normalized_market == "TW":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_tw",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_TW",
                default=default_min,
            )
        if normalized_market == "CN":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_cn",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_CN",
                default=max(default_min, CN_ACTIVE_UNIVERSE_MIN_COUNT),
            )
        if normalized_market == "CA":
            return StockUniverseService._safety_int_setting(
                field_name="asia_reconciliation_min_count_ca",
                env_name="ASIA_RECONCILIATION_MIN_COUNT_CA",
                default=max(default_min, CA_ACTIVE_UNIVERSE_MIN_COUNT),
            )
        return default_min

    @staticmethod
    def _safety_bool_setting(*, field_name: str, env_name: str, default: bool) -> bool:
        if hasattr(settings, field_name):
            return bool(getattr(settings, field_name))
        raw = (os.getenv(env_name) or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _safety_int_setting(*, field_name: str, env_name: str, default: int) -> int:
        if hasattr(settings, field_name):
            return max(0, int(getattr(settings, field_name)))
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            return max(0, int(default))
        try:
            return max(0, int(raw))
        except ValueError:
            return max(0, int(default))

    @staticmethod
    def _safety_float_setting(*, field_name: str, env_name: str, default: float) -> float:
        if hasattr(settings, field_name):
            return float(getattr(settings, field_name))
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    def _default_reconciliation_policy_for_market(
        self,
        market: str,
    ) -> UniverseReconciliationPolicy:
        normalized_market = str(market or "").strip().upper()
        return UniverseReconciliationPolicy(
            name=f"{normalized_market.lower()}_market_default",
            min_count=self._min_count_threshold_for_market(normalized_market),
            max_removed_percent=self._safety_float_setting(
                field_name="asia_reconciliation_max_removed_percent",
                env_name="ASIA_RECONCILIATION_MAX_REMOVED_PERCENT",
                default=25.0,
            ),
            anomaly_percent=self._safety_float_setting(
                field_name="asia_reconciliation_anomaly_percent",
                env_name="ASIA_RECONCILIATION_ANOMALY_PERCENT",
                default=35.0,
            ),
            apply_destructive_enabled=self._safety_bool_setting(
                field_name="asia_universe_apply_destructive_enabled",
                env_name="ASIA_UNIVERSE_APPLY_DESTRUCTIVE_ENABLED",
                default=False,
            ),
            quarantine_enforced=self._safety_bool_setting(
                field_name="asia_reconciliation_quarantine_enforced",
                env_name="ASIA_RECONCILIATION_QUARANTINE_ENFORCED",
                default=True,
            ),
        )

    def _finviz_reconciliation_policy(
        self,
        exchange_name: str | None,
    ) -> UniverseReconciliationPolicy:
        normalized_exchange = str(exchange_name or "").strip().upper() or None
        exchange_resolution = mic_alias_registry.resolve("US", normalized_exchange)
        exchange_mic = exchange_resolution.mic if exchange_resolution else None
        min_count_env = (
            "FINVIZ_RECONCILIATION_MIN_COUNT_EXCHANGE"
            if normalized_exchange
            else "FINVIZ_RECONCILIATION_MIN_COUNT_FULL"
        )
        min_count_field = (
            "finviz_reconciliation_min_count_exchange"
            if normalized_exchange
            else "finviz_reconciliation_min_count_full"
        )
        return UniverseReconciliationPolicy(
            name=(
                f"finviz_{normalized_exchange.lower()}"
                if normalized_exchange
                else "finviz_full"
            ),
            min_count=self._safety_int_setting(
                field_name=min_count_field,
                env_name=min_count_env,
                default=0 if normalized_exchange else 8000,
            ),
            max_removed_percent=self._safety_float_setting(
                field_name="finviz_reconciliation_max_removed_percent",
                env_name="FINVIZ_RECONCILIATION_MAX_REMOVED_PERCENT",
                default=10.0,
            ),
            anomaly_percent=self._safety_float_setting(
                field_name="finviz_reconciliation_anomaly_percent",
                env_name="FINVIZ_RECONCILIATION_ANOMALY_PERCENT",
                default=100.0,
            ),
            apply_destructive_enabled=self._safety_bool_setting(
                field_name="finviz_universe_apply_destructive_enabled",
                env_name="FINVIZ_UNIVERSE_APPLY_DESTRUCTIVE_ENABLED",
                default=True,
            ),
            quarantine_enforced=self._safety_bool_setting(
                field_name="finviz_reconciliation_quarantine_enforced",
                env_name="FINVIZ_RECONCILIATION_QUARANTINE_ENFORCED",
                default=True,
            ),
            removal_mics=(exchange_mic,) if exchange_mic else (),
        )

    def _evaluate_reconciliation_safety(
        self,
        *,
        market: str,
        snapshot_id: str,
        counts: Mapping[str, Any],
        removed_symbols: list[str],
        policy: UniverseReconciliationPolicy,
    ) -> dict[str, Any]:
        total_current = int(counts.get("total_current") or 0)
        total_previous = int(counts.get("total_previous") or 0)
        removed_count = int(counts.get("removed") or 0)
        changed_count = int(counts.get("changed") or 0)

        min_count_threshold = policy.min_count
        max_removed_percent = policy.max_removed_percent
        anomaly_percent_threshold = policy.anomaly_percent

        removed_percent = (removed_count / total_previous * 100.0) if total_previous > 0 else 0.0
        anomaly_percent = (
            (removed_count + changed_count) / total_previous * 100.0
            if total_previous > 0
            else 0.0
        )

        gate_breaches: list[dict[str, Any]] = []
        if total_current < min_count_threshold:
            gate_breaches.append(
                {
                    "gate": "min_count",
                    "actual": total_current,
                    "threshold": min_count_threshold,
                    "comparator": ">=",
                }
            )
        if total_previous > 0 and removed_percent > max_removed_percent:
            gate_breaches.append(
                {
                    "gate": "max_removed_percent",
                    "actual": round(removed_percent, 4),
                    "threshold": round(max_removed_percent, 4),
                    "comparator": "<=",
                }
            )
        if total_previous > 0 and anomaly_percent > anomaly_percent_threshold:
            gate_breaches.append(
                {
                    "gate": "anomaly_percent",
                    "actual": round(anomaly_percent, 4),
                    "threshold": round(anomaly_percent_threshold, 4),
                    "comparator": "<=",
                }
            )

        apply_destructive_enabled = policy.apply_destructive_enabled
        quarantine_enforced = policy.quarantine_enforced
        quarantined = bool(gate_breaches) and quarantine_enforced
        allow_destructive_apply = apply_destructive_enabled and not quarantined
        destructive_apply_blocked = bool(removed_symbols) and not allow_destructive_apply

        alerts: list[str] = []
        if gate_breaches:
            alerts.append(
                f"{market} snapshot {snapshot_id} breached reconciliation safety thresholds"
            )
        if destructive_apply_blocked and not apply_destructive_enabled:
            alerts.append(
                "Destructive apply disabled by asia_universe_apply_destructive_enabled=false"
            )
        if destructive_apply_blocked and quarantined:
            alerts.append(
                "Destructive apply blocked by enforced reconciliation quarantine"
            )

        return {
            "policy_name": policy.name,
            "quarantined": quarantined,
            "apply_destructive_enabled": apply_destructive_enabled,
            "quarantine_enforced": quarantine_enforced,
            "removal_mics": list(policy.removal_mics),
            "allow_destructive_apply": allow_destructive_apply,
            "destructive_apply_blocked": destructive_apply_blocked,
            "gate_breaches": gate_breaches,
            "thresholds": {
                "min_count": min_count_threshold,
                "max_removed_percent": max_removed_percent,
                "anomaly_percent": anomaly_percent_threshold,
            },
            "metrics": {
                "total_current": total_current,
                "total_previous": total_previous,
                "removed_percent": round(removed_percent, 4),
                "anomaly_percent": round(anomaly_percent, 4),
            },
            "alerts": alerts,
        }

    def _apply_market_reconciliation_policy(
        self,
        db: Session,
        *,
        market: str,
        snapshot_id: str,
        trigger_source: str,
        reconciliation_policy: UniverseReconciliationPolicy | None = None,
        reconciliation: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        run_id = int(reconciliation.get("run_id") or 0)
        run = (
            db.query(StockUniverseReconciliationRun)
            .filter(StockUniverseReconciliationRun.id == run_id)
            .one_or_none()
        )
        if run is None:
            raise ValueError(f"Missing reconciliation run {run_id} for {market}:{snapshot_id}")

        artifact_payload = json.loads(run.artifact_json or "{}")
        removed_symbols_full = sorted(
            {
                str(symbol).strip().upper()
                for symbol in artifact_payload.get("removed_symbols", [])
                if str(symbol).strip()
            }
        )
        safety = self._evaluate_reconciliation_safety(
            market=market,
            snapshot_id=snapshot_id,
            counts=reconciliation.get("counts") or {},
            removed_symbols=removed_symbols_full,
            policy=(
                reconciliation_policy
                or self._default_reconciliation_policy_for_market(market)
            ),
        )

        deactivated_symbols: list[str] = []
        if safety["allow_destructive_apply"] and removed_symbols_full:
            candidates_query = (
                db.query(StockUniverse)
                .filter(
                    StockUniverse.market == market,
                    StockUniverse.symbol.in_(removed_symbols_full),
                )
            )
            removal_mics = tuple(safety.get("removal_mics") or ())
            if removal_mics:
                candidates_query = candidates_query.filter(
                    StockUniverse.exchange.in_(removal_mics)
                )
            candidates = candidates_query.all()
            for record in candidates:
                if self._normalize_status(record) != UNIVERSE_STATUS_ACTIVE:
                    continue
                changed = self._apply_status_transition(
                    db,
                    record,
                    new_status=UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
                    trigger_source=f"{trigger_source}_reconcile_apply",
                    reason=f"Missing from {market} source snapshot {snapshot_id}",
                    now=now,
                    payload={
                        "snapshot_id": snapshot_id,
                        "reconciliation_run_id": run_id,
                        "market": market,
                    },
                )
                if changed:
                    deactivated_symbols.append(record.symbol)

        if safety["alerts"]:
            logger.warning(
                "UNIVERSE SAFETY ALERT market=%s snapshot=%s run_id=%s alerts=%s breaches=%s",
                market,
                snapshot_id,
                run_id,
                safety["alerts"],
                safety["gate_breaches"],
            )

        details_limit = 25
        safety["deactivated_count"] = len(deactivated_symbols)
        safety["deactivated_symbols"] = sorted(deactivated_symbols)[:details_limit]
        safety["deactivated_symbols_truncated"] = len(deactivated_symbols) > details_limit

        artifact_payload["safety"] = safety
        run.artifact_json = self._stable_json(artifact_payload)
        run.artifact_hash = self._sha256_text(run.artifact_json)
        db.flush()

        updated_reconciliation = dict(reconciliation)
        updated_reconciliation["artifact_hash"] = run.artifact_hash
        updated_reconciliation["safety"] = safety
        return updated_reconciliation

    def _ingest_snapshot_rows_via_pipeline(
        self,
        db: Session,
        *,
        market: str,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
        ingestion_context: UniverseIngestionContext | None = None,
    ) -> Dict[str, Any]:
        return self._universe_ingestion_pipeline.ingest_snapshot_rows(
            db,
            market=market,
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
            ingestion_context=(
                ingestion_context
                or UniverseIngestionContext.default_for_market(
                    market,
                    reconciliation_policy=self._default_reconciliation_policy_for_market(market),
                )
            ),
        )

    def _deactivate_india_coverage_rejections(
        self,
        db: Session,
        context: UniverseBeforeReconciliationContext,
        *,
        now: datetime,
    ) -> dict[str, int]:
        rows = context.result.side_effects.coverage_rejections
        if not rows:
            return {"coverage_rejected": 0}

        existing_rows = {
            row.symbol: row
            for row in db.query(StockUniverse)
            .filter(StockUniverse.symbol.in_([row.symbol for row in rows]))
            .all()
        }
        for coverage_rejection in rows:
            rejected_row = coverage_rejection.rejected_row
            existing = existing_rows.get(coverage_rejection.symbol)
            if existing is None or self._normalize_status(existing) != UNIVERSE_STATUS_ACTIVE:
                continue
            self._apply_status_transition(
                db,
                existing,
                new_status=UNIVERSE_STATUS_INACTIVE_NO_DATA,
                trigger_source="in_ingest_coverage_gate",
                reason=f"Rejected by IN BSE coverage gate: {rejected_row.reason}",
                now=now,
                payload={
                    "source_name": rejected_row.source_name or context.source_name,
                    "source_symbol": rejected_row.source_symbol,
                    "source_row_number": rejected_row.source_row_number,
                    "snapshot_id": rejected_row.snapshot_id or context.snapshot_id,
                    "snapshot_as_of": (
                        str(rejected_row.snapshot_as_of)
                        if rejected_row.snapshot_as_of is not None
                        else None
                    ),
                    "symbol": coverage_rejection.symbol,
                    "reason": rejected_row.reason,
                },
                source="in_ingest",
            )
        return {"coverage_rejected": len(rows)}

    def ingest_in_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest IN rows with exchange-aware canonicalization and lineage metadata."""
        source_canonical_rows, source_rejected_rows = self._canonicalize_in_snapshot_rows(
            rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
        )

        if strict and source_rejected_rows:
            sample = "; ".join(
                f"row {row.source_row_number}: {row.reason}"
                for row in source_rejected_rows[:3]
            )
            raise ValueError(
                f"IN ingestion rejected {len(source_rejected_rows)} row(s). {sample}"
            )

        accepted_rows, coverage_rejected_rows = self._filter_india_bse_rows_for_downstream_support(
            db,
            source_canonical_rows,
        )

        source_rejections = tuple(
            FlatUniverseCanonicalizerAdapter.rejected_row_from_flat(row)
            for row in source_rejected_rows
        )
        coverage_rejections: list[RejectedUniverseRow] = []
        coverage_side_effects: list[UniverseCoverageRejection] = []
        for row in coverage_rejected_rows:
            rejected_row = RejectedUniverseRow(
                source_row_number=row.source_row_number,
                source_symbol=row.source_symbol,
                reason=row.reason,
                source_name=source_name,
                snapshot_id=snapshot_id,
                snapshot_as_of=snapshot_as_of,
                strict=False,
            )
            coverage_rejections.append(rejected_row)
            coverage_side_effects.append(
                UniverseCoverageRejection(
                    symbol=row.symbol,
                    rejected_row=rejected_row,
                )
            )

        result = CanonicalUniverseIngestionResult(
            canonical_rows=tuple(
                FlatUniverseCanonicalizerAdapter.canonical_row_from_flat(row)
                for row in accepted_rows
            ),
            rejected_rows=source_rejections + tuple(coverage_rejections),
            side_effects=UniverseIngestionSideEffects(
                coverage_rejections=tuple(coverage_side_effects),
            ),
        )
        return self._universe_ingestion_pipeline.ingest_canonicalized_result(
            db,
            market="IN",
            source_name=source_name,
            snapshot_id=snapshot_id,
            result=result,
            strict=strict,
            ingestion_context=UniverseIngestionContext.default_for_market(
                "IN",
                reconciliation_policy=self._default_reconciliation_policy_for_market("IN"),
            ),
        )

    def ingest_au_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest AU rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="AU",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_hk_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse HK ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_hk_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest HK rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="HK",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_hk_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest HK universe rows from CSV content."""
        rows = self._parse_hk_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("hk")
        return self.ingest_hk_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_jp_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse JP ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_jp_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest JP rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="JP",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_jp_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest JP universe rows from CSV content."""
        rows = self._parse_jp_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("jp")
        return self.ingest_jp_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_kr_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse KR ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_kr_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest KR rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="KR",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_kr_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest KR universe rows from CSV content."""
        rows = self._parse_kr_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("kr")
        return self.ingest_kr_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_cn_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse CN ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker", "code"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=[
                    "symbol",
                    "name",
                    "exchange",
                    "sector",
                    "industry_group",
                    "industry",
                    "sub_industry",
                    "market_cap",
                ],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(
                    row_lower.get("symbol")
                    or row_lower.get("local_code")
                    or row_lower.get("ticker")
                    or row_lower.get("code")
                    or ""
                )
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER", "CODE"}:
                continue
            rows.append(row_lower)
        return rows

    def _upsert_cn_stock_industry_rows(
        self,
        db: Session,
        rows: Iterable[Any],
        *,
        now: datetime,
    ) -> int:
        """Populate stock_industry with CN sector/group/industry/sub-industry data."""
        canonical_rows = [
            row
            for row in rows
            if any(
                str(getattr(row, attr, "") or "").strip()
                for attr in ("sector", "industry_group", "industry", "sub_industry")
            )
        ]
        if not canonical_rows:
            return 0

        symbols = [row.symbol for row in canonical_rows]
        existing_by_symbol = {
            record.symbol: record
            for record in db.query(StockIndustry).filter(StockIndustry.symbol.in_(symbols)).all()
        }
        new_rows: list[StockIndustry] = []
        changed = 0
        for row in canonical_rows:
            existing = existing_by_symbol.get(row.symbol)
            if existing is not None:
                if row.sector:
                    existing.sector = row.sector
                if row.industry_group:
                    existing.industry_group = row.industry_group
                if row.industry:
                    existing.industry = row.industry
                if row.sub_industry:
                    existing.sub_industry = row.sub_industry
                existing.updated_at = now
                changed += 1
                continue
            new_rows.append(
                StockIndustry(
                    symbol=row.symbol,
                    sector=row.sector or None,
                    industry_group=row.industry_group or None,
                    industry=row.industry or row.industry_group or None,
                    sub_industry=row.sub_industry or None,
                    updated_at=now,
                )
            )
            changed += 1

        self._bulk_insert_records(db, new_rows)
        return changed

    def _upsert_cn_stock_industry_from_pipeline_context(
        self,
        db: Session,
        context: UniverseBeforeReconciliationContext,
        *,
        now: datetime,
    ) -> dict[str, int]:
        return {
            "stock_industry_upserts": self._upsert_cn_stock_industry_rows(
                db,
                context.result.side_effects.industry_taxonomy_rows,
                now=now,
            )
        }

    def ingest_cn_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest CN rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="CN",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_cn_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest CN universe rows from CSV content."""
        rows = self._parse_cn_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("cn")
        return self.ingest_cn_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_tw_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse TW ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_tw_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest TW rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="TW",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_tw_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest TW universe rows from CSV content."""
        rows = self._parse_tw_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("tw")
        return self.ingest_tw_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_ca_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse CA ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_ca_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest CA rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="CA",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_ca_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest CA universe rows from CSV content."""
        rows = self._parse_ca_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("ca")
        return self.ingest_ca_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    @staticmethod
    def _parse_de_csv_rows(csv_content: str) -> list[dict[str, Any]]:
        """Parse DE ingestion CSV into normalized lowercase-key row dicts."""
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(name).strip().lower() for name in fieldnames if name]
        has_symbol_field = any(name in {"symbol", "local_code", "ticker"} for name in normalized_fields)
        if not fieldnames or not has_symbol_field:
            csv_file.seek(0)
            reader = csv.DictReader(
                csv_file,
                fieldnames=["symbol", "name", "exchange", "sector", "industry", "market_cap"],
            )

        rows: list[dict[str, Any]] = []
        for row in reader:
            row_lower = {str(k).strip().lower(): v for k, v in row.items() if k}
            source_symbol = (
                str(row_lower.get("symbol") or row_lower.get("local_code") or row_lower.get("ticker") or "")
                .strip()
                .upper()
            )
            if not source_symbol or source_symbol in {"SYMBOL", "LOCAL_CODE", "TICKER"}:
                continue
            rows.append(row_lower)
        return rows

    def ingest_de_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest DE rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="DE",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_de_from_csv(
        self,
        db: Session,
        csv_content: str,
        *,
        source_name: str,
        snapshot_id: str | None = None,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest DE universe rows from CSV content."""
        rows = self._parse_de_csv_rows(csv_content)
        resolved_snapshot_id = snapshot_id or self._auto_snapshot_id("de")
        return self.ingest_de_snapshot_rows(
            db,
            rows=rows,
            source_name=source_name,
            snapshot_id=resolved_snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_sg_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest SG rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="SG",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def ingest_my_snapshot_rows(
        self,
        db: Session,
        *,
        rows: Iterable[dict[str, Any]],
        source_name: str,
        snapshot_id: str,
        snapshot_as_of: str | None = None,
        source_metadata: Optional[dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Ingest MY rows with deterministic canonicalization and lineage metadata."""
        return self._ingest_snapshot_rows_via_pipeline(
            db,
            market="MY",
            rows=rows,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )

    def populate_from_csv(self, db: Session, csv_content: str) -> Dict:
        """
        Populate stock_universe table from CSV file.

        Args:
            db: Database session
            csv_content: CSV file content as string

        Returns:
            Dict with stats: {added: int, updated: int, total: int}
        """
        try:
            # Import stocks from CSV
            stocks = self.import_from_csv(csv_content)

            if not stocks:
                logger.warning("No stocks imported from CSV")
                return {'added': 0, 'updated': 0, 'total': 0}

            added_count = 0
            updated_count = 0
            now = datetime.utcnow()
            new_rows: list[StockUniverse] = []
            new_events: list[StockUniverseStatusEvent] = []
            resolved_rows: list[tuple[dict[str, Any], Any, str, str]] = []
            lookup_symbols: set[str] = set()
            for stock_data in stocks:
                identity = self._resolved_identity(stock_data)
                source_symbol = stock_data["symbol"]
                canonical_symbol = identity.canonical_symbol
                resolved_rows.append((stock_data, identity, source_symbol, canonical_symbol))
                lookup_symbols.add(source_symbol)
                lookup_symbols.add(canonical_symbol)

            stock_map = {
                stock.symbol: stock
                for stock in db.query(StockUniverse).filter(
                    StockUniverse.symbol.in_(list(lookup_symbols))
                ).all()
            }

            for stock_data, identity, source_symbol, canonical_symbol in resolved_rows:
                existing = stock_map.get(canonical_symbol) or stock_map.get(source_symbol)
                if existing and existing.symbol != canonical_symbol and canonical_symbol not in stock_map:
                    stock_map.pop(existing.symbol, None)
                    existing.symbol = canonical_symbol
                    stock_map[canonical_symbol] = existing

                if existing:
                    existing.name = stock_data["name"] or existing.name
                    existing.exchange = identity.exchange or existing.exchange
                    existing.market = identity.market
                    existing.currency = identity.currency
                    existing.timezone = identity.timezone
                    existing.local_code = identity.local_code or existing.local_code
                    existing.sector = prefer_meaningful(stock_data["sector"], existing.sector)
                    existing.industry = prefer_meaningful(stock_data["industry"], existing.industry)
                    existing.market_cap = stock_data["market_cap"] or existing.market_cap
                    self._apply_status_transition(
                        db,
                        existing,
                        new_status=UNIVERSE_STATUS_ACTIVE,
                        trigger_source="csv_import",
                        reason="Imported from CSV",
                        now=now,
                        payload={"source": "csv"},
                        source="csv",
                        clear_failures=True,
                    )
                    updated_count += 1
                else:
                    new_rows.append(StockUniverse(
                        symbol=canonical_symbol,
                        name=stock_data["name"],
                        market=identity.market,
                        exchange=identity.exchange,
                        currency=identity.currency,
                        timezone=identity.timezone,
                        local_code=identity.local_code,
                        sector=stock_data["sector"],
                        industry=stock_data["industry"],
                        market_cap=stock_data["market_cap"],
                        is_active=True,
                        status=UNIVERSE_STATUS_ACTIVE,
                        status_reason="Imported from CSV",
                        source="csv",
                        consecutive_fetch_failures=0,
                        added_at=now,
                        first_seen_at=now,
                        updated_at=now,
                    ))
                    new_events.append(self._build_status_event_record(
                        symbol=canonical_symbol,
                        old_status=None,
                        new_status=UNIVERSE_STATUS_ACTIVE,
                        trigger_source="csv_import",
                        reason="Imported from CSV",
                        payload={"source": "csv"},
                    ))
                    added_count += 1

            self._bulk_insert_records(db, new_rows)
            self._bulk_insert_records(db, new_events)
            db.commit()

            logger.info(f"CSV import completed: {added_count} added, {updated_count} updated")

            return {
                'added': added_count,
                'updated': updated_count,
                'total': len(stocks),
            }

        except Exception as e:
            logger.error(f"Error populating from CSV: {e}", exc_info=True)
            db.rollback()
            raise

    def _parse_market_cap(self, market_cap_str: str) -> Optional[float]:
        """
        Parse market cap string from finviz (e.g., '2.5B', '500M').

        Returns:
            Market cap in dollars or None
        """
        if not market_cap_str or market_cap_str == '-':
            return None

        try:
            market_cap_str = market_cap_str.strip().upper()
            multiplier = 1

            if market_cap_str.endswith('B'):
                multiplier = 1e9
                market_cap_str = market_cap_str[:-1]
            elif market_cap_str.endswith('M'):
                multiplier = 1e6
                market_cap_str = market_cap_str[:-1]
            elif market_cap_str.endswith('K'):
                multiplier = 1e3
                market_cap_str = market_cap_str[:-1]

            value = float(market_cap_str)
            return value * multiplier

        except (ValueError, AttributeError):
            return None

    def _determine_exchange(self, row: Dict) -> str:
        """
        Determine exchange from finviz row data.

        Args:
            row: Finviz data row

        Returns:
            Exchange code: NYSE, NASDAQ, or AMEX
        """
        # Try to get from row if available
        # Finviz doesn't always provide exchange directly, so we may need to infer
        # For now, default to NASDAQ (most tech stocks) unless we can determine otherwise
        # This is a limitation - may need enhancement
        return row.get('Exchange', 'NASDAQ').upper()

    def populate_universe(self, db: Session, exchange_filter: Optional[str] = None) -> Dict:
        """
        Populate stock_universe table from finviz.

        Performs shared upsert/reconciliation using the canonical Universe pipeline.

        Args:
            db: Database session
            exchange_filter: Optional exchange filter

        Returns:
            Dict with stats: {added: int, updated: int, deactivated: int, total: int}
        """
        try:
            # Fetch stocks from finviz
            stocks = self.fetch_from_finviz(exchange_filter)

            if not stocks:
                logger.warning("No stocks fetched from finviz")
                return {'added': 0, 'updated': 0, 'deactivated': 0, 'total': 0}

            exchange_name = str(exchange_filter or "").strip().upper() or None
            source_name = (
                "finviz"
                if exchange_name is None
                else f"finviz_{exchange_name.lower()}"
            )
            stats = self._ingest_snapshot_rows_via_pipeline(
                db,
                market="US",
                rows=stocks,
                source_name=source_name,
                snapshot_id=self._auto_snapshot_id(source_name),
                source_metadata={"exchange_filter": exchange_name},
                strict=True,
                ingestion_context=UniverseIngestionContext(
                    trigger_source="finviz_sync",
                    row_source="finviz",
                    reconciliation_policy=self._finviz_reconciliation_policy(exchange_name),
                ),
            )
            reconciliation = stats.get("reconciliation") or {}
            safety = reconciliation.get("safety") or {}
            stats["deactivated"] = int(safety.get("deactivated_count") or 0)

            logger.info(
                "Universe populated: %s added, %s updated, %s deactivated",
                stats["added"],
                stats["updated"],
                stats["deactivated"],
            )
            return stats

        except Exception as e:
            logger.error(f"Error populating universe: {e}", exc_info=True)
            db.rollback()
            raise

    def get_active_symbols(
        self,
        db: Session,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        listing_tier: Optional[str] = None,
        sector: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        sp500_only: bool = False,
        index_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get list of active symbols for scanning.

        Args:
            db: Database session
            market: Optional market filter (US, HK, JP, TW)
            exchange: Optional exchange filter (NYSE, NASDAQ, AMEX)
            listing_tier: Optional canonical listing tier filter
            sector: Optional sector filter
            min_market_cap: Optional minimum market cap filter
            sp500_only: If True, only return S&P 500 stocks (legacy path;
                equivalent to ``index_name="SP500"``)
            index_name: Optional index membership filter. ``"SP500"`` maps to
                the legacy ``is_sp500`` column; any other value resolves via
                ``stock_universe_index_membership``. Empty / unknown indices
                return no rows (fail-closed so an unseeded index doesn't leak
                a whole-market scan).
            limit: Optional limit on number of symbols

        Returns:
            List of symbol strings
        """
        try:
            query = db.query(StockUniverse.symbol).filter(
                StockUniverse.active_filter()
            )

            resolved_index = index_name.upper() if index_name else None
            if sp500_only:
                resolved_index = IndexName.SP500.value

            if resolved_index == IndexName.SP500.value:
                query = query.filter(StockUniverse.is_sp500 == True)
            elif resolved_index:
                known_indices = {entry.value for entry in IndexName}
                if resolved_index not in known_indices:
                    # Typo or a future index not yet in the enum. Fail-closed
                    # (return []) but surface the miss as a warning so ops
                    # can't mistake an unseeded index for an empty market.
                    logger.warning(
                        "get_active_symbols called with unknown index_name=%s; "
                        "returning [] (known: %s)",
                        resolved_index,
                        sorted(known_indices),
                    )
                    return []
                membership_symbols = db.query(
                    StockUniverseIndexMembership.symbol
                ).filter(StockUniverseIndexMembership.index_name == resolved_index)
                membership_count = membership_symbols.count()
                if membership_count == 0:
                    logger.warning(
                        "get_active_symbols: index_name=%s is a known index but has "
                        "no seeded membership rows — did you forget to run the seed "
                        "script? Returning [].",
                        resolved_index,
                    )
                query = query.filter(StockUniverse.symbol.in_(membership_symbols))

            if market:
                normalized_market = market.upper()
                fallback_exchanges = MARKET_EXCHANGE_FALLBACKS.get(normalized_market)
                if fallback_exchanges:
                    query = query.filter(
                        or_(
                            StockUniverse.market == normalized_market,
                            and_(
                                or_(
                                    StockUniverse.market.is_(None),
                                    func.trim(StockUniverse.market) == "",
                                ),
                                StockUniverse.exchange.in_(fallback_exchanges),
                            ),
                        )
                    )
                else:
                    query = query.filter(StockUniverse.market == normalized_market)

            if exchange:
                query = query.filter(
                    StockUniverse.exchange.in_(
                        self._exchange_filter_values(market, exchange)
                    )
                )

            if listing_tier:
                query = query.filter(StockUniverse.listing_tier == listing_tier)

            if sector:
                query = query.filter(StockUniverse.sector == sector)

            if min_market_cap is not None:
                query = query.filter(StockUniverse.market_cap >= min_market_cap)

            # Order by market cap descending (scan large caps first)
            query = query.order_by(StockUniverse.market_cap.desc())

            if limit:
                query = query.limit(limit)

            symbols = [row[0] for row in query.all()]

            logger.info(
                "Retrieved %d active symbols (market=%s, exchange=%s, tier=%s, sector=%s, index=%s)",
                len(symbols),
                market,
                exchange,
                listing_tier,
                sector,
                resolved_index,
            )
            return symbols

        except Exception as e:
            logger.error(f"Error getting active symbols: {e}", exc_info=True)
            return []

    @staticmethod
    def _exchange_filter_values(
        market: Optional[str],
        exchange: str,
    ) -> tuple[str, ...]:
        requested = str(exchange or "").strip().upper()
        if not requested:
            return ()

        market_code = str(market or "").strip().upper() or None
        resolved = (
            mic_alias_registry.resolve(market_code, requested)
            if market_code
            else mic_alias_registry.resolve_global(requested)
        )
        if resolved is None:
            return (requested,)

        values = (
            *mic_alias_registry.aliases_for_mic(resolved.market, resolved.mic),
            requested,
            resolved.mic,
        )
        return tuple(dict.fromkeys(value for value in values if value))

    def add_manual_symbol(self, db: Session, symbol: str, name: str = "") -> bool:
        """
        Manually add a symbol to the universe.

        Args:
            db: Database session
            symbol: Stock symbol
            name: Company name

        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()

            # Check if already exists
            existing = db.query(StockUniverse).filter(StockUniverse.symbol == symbol).first()

            if existing:
                # Reactivate if inactive
                if self._normalize_status(existing) != UNIVERSE_STATUS_ACTIVE:
                    existing.name = name or existing.name
                    if existing.exchange is None:
                        existing.exchange = 'MANUAL'
                    self._apply_status_transition(
                        db,
                        existing,
                        new_status=UNIVERSE_STATUS_ACTIVE,
                        trigger_source="manual_add",
                        reason="Manually reactivated by admin",
                        now=datetime.utcnow(),
                        payload={"source": "manual"},
                        source="manual",
                        clear_failures=True,
                    )
                    db.commit()
                    logger.info(f"Reactivated symbol: {symbol}")
                    return True
                else:
                    logger.info(f"Symbol already exists and is active: {symbol}")
                    return True
            else:
                # Add new symbol
                new_stock = StockUniverse(
                    symbol=symbol,
                    name=name,
                    exchange='MANUAL',
                    is_active=True,
                    status=UNIVERSE_STATUS_ACTIVE,
                    status_reason="Manually added by admin",
                    source='manual',
                    added_at=datetime.utcnow(),
                    first_seen_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(new_stock)
                self._add_status_event(
                    db,
                    symbol=symbol,
                    old_status=None,
                    new_status=UNIVERSE_STATUS_ACTIVE,
                    trigger_source="manual_add",
                    reason="Manually added by admin",
                    payload={"source": "manual"},
                )
                db.commit()
                logger.info(f"Added manual symbol: {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error adding manual symbol {symbol}: {e}", exc_info=True)
            db.rollback()
            return False

    def deactivate_symbol(self, db: Session, symbol: str) -> bool:
        """
        Deactivate a symbol (remove from scanning).

        Args:
            db: Database session
            symbol: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = symbol.upper()

            stock = db.query(StockUniverse).filter(StockUniverse.symbol == symbol).first()

            if stock:
                self._apply_status_transition(
                    db,
                    stock,
                    new_status=UNIVERSE_STATUS_INACTIVE_MANUAL,
                    trigger_source="manual_deactivate",
                    reason="Manually deactivated by admin",
                    now=datetime.utcnow(),
                    payload={"source": "manual"},
                )
                db.commit()
                logger.info(f"Deactivated symbol: {symbol}")
                return True
            else:
                logger.warning(f"Symbol not found: {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error deactivating symbol {symbol}: {e}", exc_info=True)
            db.rollback()
            return False

    def get_stats(self, db: Session) -> Dict:
        """
        Get universe statistics.

        Returns:
            Dict with total, active, by_exchange counts
        """
        try:
            total = db.query(StockUniverse).count()
            active = db.query(StockUniverse).filter(
                StockUniverse.active_filter()
            ).count()
            market_audit = self.get_market_audit(db)

            # Count by exchange
            by_exchange = {}
            exchanges = db.query(
                StockUniverse.exchange,
                func.count(StockUniverse.id),
            ).filter(
                StockUniverse.active_filter()
            ).group_by(StockUniverse.exchange).all()

            for exchange, count in exchanges:
                by_exchange[exchange] = count

            # Count S&P 500 members
            sp500 = db.query(StockUniverse).filter(
                StockUniverse.active_filter(),
                StockUniverse.is_sp500 == True,
            ).count()

            by_status: Dict[str, int] = {}
            for record in db.query(StockUniverse).all():
                normalized = self._normalize_status(record)
                by_status[normalized] = by_status.get(normalized, 0) + 1
            recent_deactivations = db.query(StockUniverseStatusEvent).filter(
                StockUniverseStatusEvent.new_status != UNIVERSE_STATUS_ACTIVE
            ).order_by(StockUniverseStatusEvent.created_at.desc()).limit(10).all()

            return {
                'total': total,
                'active': active,
                'by_exchange': by_exchange,
                'sp500': sp500,
                'by_status': by_status,
                'by_market': market_audit.get('by_market', {}),
                'market_checks': market_audit.get('checks', {}),
                'recent_deactivations': [
                    {
                        'symbol': event.symbol,
                        'new_status': event.new_status,
                        'reason': event.reason,
                        'created_at': event.created_at.isoformat() if event.created_at else None,
                    }
                    for event in recent_deactivations
                ],
            }

        except Exception as e:
            logger.error(f"Error getting universe stats: {e}", exc_info=True)
            return {
                'total': 0,
                'active': 0,
                'by_exchange': {},
                'sp500': 0,
                'by_status': {},
                'by_market': {},
                'market_checks': {
                    'stale_after_hours': 0,
                    'stale_markets': [],
                    'quarantined_markets': [],
                    'missing_snapshot_markets': [],
                    'has_stale_markets': False,
                    'has_quarantined_markets': False,
                },
                'recent_deactivations': [],
            }

    def get_market_audit(self, db: Session) -> Dict[str, Any]:
        """
        Return market-level universe audit with freshness + reconciliation summary.

        The payload is designed for ops visibility and launch-gate checks:
        - counts by market + normalized lifecycle status
        - latest reconciliation snapshot age/diff summary for official universe markets
        - stale/missing/quarantine check lists
        """
        now = datetime.utcnow()
        stale_hours_raw = os.getenv(
            "ASIA_UNIVERSE_AUDIT_STALE_HOURS",
            str(getattr(settings, "asia_universe_audit_stale_hours", 36) or 36),
        )
        try:
            configured_stale_hours = int(stale_hours_raw)
        except (TypeError, ValueError):
            configured_stale_hours = 36
        stale_after_hours = max(
            configured_stale_hours,
            1,
        )
        stale_after_seconds = stale_after_hours * 3600
        reconciliation_markets = set(
            _MARKET_CATALOG.market_codes_with_capability("official_universe")
        )

        by_market: Dict[str, Dict[str, Any]] = {
            market: {
                "market": market,
                "counts": {"total": 0, "active": 0, "inactive": 0},
                "by_status": {},
                "latest_seen_in_source_at": None,
                "latest_snapshot": None,
            }
            for market in _MARKET_CATALOG.supported_market_codes()
        }

        universe_rows = db.query(
            StockUniverse.market,
            StockUniverse.is_active,
            StockUniverse.status,
            StockUniverse.last_seen_in_source_at,
        ).all()
        for market, is_active, raw_status, seen_at in universe_rows:
            market_code = (market or "").strip().upper() or "UNKNOWN"
            if market_code not in by_market:
                by_market[market_code] = {
                    "market": market_code,
                    "counts": {"total": 0, "active": 0, "inactive": 0},
                    "by_status": {},
                    "latest_seen_in_source_at": None,
                    "latest_snapshot": None,
                }
            entry = by_market[market_code]
            entry["counts"]["total"] += 1
            if bool(is_active):
                entry["counts"]["active"] += 1
            else:
                entry["counts"]["inactive"] += 1

            normalized_status = self._normalize_status_fields(raw_status, is_active)
            entry["by_status"][normalized_status] = entry["by_status"].get(normalized_status, 0) + 1

            last_seen = entry["latest_seen_in_source_at"]
            if seen_at is not None and (last_seen is None or seen_at > last_seen):
                entry["latest_seen_in_source_at"] = seen_at

        latest_runs_subquery = (
            db.query(StockUniverseReconciliationRun)
            .with_entities(
                StockUniverseReconciliationRun.market.label("market"),
                func.max(StockUniverseReconciliationRun.created_at).label("max_created_at"),
            )
            .group_by(StockUniverseReconciliationRun.market)
            .subquery()
        )
        latest_runs = (
            db.query(StockUniverseReconciliationRun)
            .join(
                latest_runs_subquery,
                and_(
                    StockUniverseReconciliationRun.market == latest_runs_subquery.c.market,
                    StockUniverseReconciliationRun.created_at == latest_runs_subquery.c.max_created_at,
                ),
            )
            .order_by(
                StockUniverseReconciliationRun.market.asc(),
                StockUniverseReconciliationRun.id.desc(),
            )
            .all()
        )
        latest_by_market: dict[str, StockUniverseReconciliationRun] = {}
        for run in latest_runs:
            market_code = (run.market or "").strip().upper()
            if market_code and market_code not in latest_by_market:
                latest_by_market[market_code] = run

        stale_markets: list[str] = []
        quarantined_markets: list[str] = []
        missing_snapshot_markets: list[str] = []

        for market_code in sorted(set(by_market) | set(latest_by_market)):
            if market_code not in by_market:
                by_market[market_code] = {
                    "market": market_code,
                    "counts": {"total": 0, "active": 0, "inactive": 0},
                    "by_status": {},
                    "latest_seen_in_source_at": None,
                    "latest_snapshot": None,
                }
            entry = by_market[market_code]
            run = latest_by_market.get(market_code)
            supports_snapshot = market_code in reconciliation_markets

            latest_snapshot: Dict[str, Any] | None = None
            if run is not None:
                artifact_payload: dict[str, Any] = {}
                if run.artifact_json:
                    try:
                        parsed = json.loads(run.artifact_json)
                        if isinstance(parsed, dict):
                            artifact_payload = parsed
                    except Exception:
                        logger.warning(
                            "Failed to parse reconciliation artifact for market audit",
                            exc_info=True,
                            extra={"market": market_code, "snapshot_id": run.snapshot_id},
                        )
                safety = artifact_payload.get("safety") if isinstance(artifact_payload.get("safety"), dict) else {}
                age_seconds = self._snapshot_age_seconds(now=now, value=run.created_at)
                is_stale = bool(
                    supports_snapshot and age_seconds is not None and age_seconds > stale_after_seconds
                )
                latest_snapshot = {
                    "snapshot_id": run.snapshot_id,
                    "previous_snapshot_id": run.previous_snapshot_id,
                    "created_at": self._utc_iso(run.created_at),
                    "age_seconds": age_seconds,
                    "is_stale": is_stale,
                    "counts": {
                        "total_current": int(run.total_current or 0),
                        "total_previous": int(run.total_previous or 0),
                        "added": int(run.added_count or 0),
                        "removed": int(run.removed_count or 0),
                        "changed": int(run.changed_count or 0),
                        "unchanged": int(run.unchanged_count or 0),
                    },
                    "safety": {
                        "quarantined": bool(safety.get("quarantined")),
                        "destructive_apply_blocked": bool(safety.get("destructive_apply_blocked")),
                        "gate_breaches": safety.get("gate_breaches", []),
                        "alerts": safety.get("alerts", []),
                    },
                }
                if latest_snapshot["is_stale"]:
                    stale_markets.append(market_code)
                if latest_snapshot["safety"]["quarantined"]:
                    quarantined_markets.append(market_code)
            elif supports_snapshot and entry["counts"]["total"] > 0:
                stale_markets.append(market_code)
                missing_snapshot_markets.append(market_code)

            entry["latest_seen_in_source_at"] = self._utc_iso(entry["latest_seen_in_source_at"])
            entry["latest_snapshot"] = latest_snapshot
            entry["snapshot_supported"] = supports_snapshot

        return {
            "generated_at": self._utc_iso(now),
            "by_market": by_market,
            "checks": {
                "stale_after_hours": stale_after_hours,
                "stale_markets": sorted(set(stale_markets)),
                "quarantined_markets": sorted(set(quarantined_markets)),
                "missing_snapshot_markets": sorted(set(missing_snapshot_markets)),
                "has_stale_markets": bool(stale_markets),
                "has_quarantined_markets": bool(quarantined_markets),
            },
        }

    def filter_active_symbols(
        self,
        db: Session,
        symbols: Iterable[str],
    ) -> List[str]:
        """Return the active subset of the supplied symbols in input order."""
        ordered = [symbol.upper() for symbol in symbols]
        if not ordered:
            return []

        active_set = {
            row[0]
            for row in db.query(StockUniverse.symbol).filter(
                StockUniverse.symbol.in_(ordered),
                StockUniverse.active_filter(),
            ).all()
        }
        return [symbol for symbol in ordered if symbol in active_set]

    def get_active_symbol_set(self, db: Session) -> set[str]:
        """Return the set of currently active universe symbols."""
        return {
            row[0]
            for row in db.query(StockUniverse.symbol).filter(
                StockUniverse.active_filter()
            ).all()
        }

    def record_fetch_success(self, db: Session, symbol: str) -> bool:
        """Reset failure counters after a successful provider fetch."""
        record = db.query(StockUniverse).filter(
            StockUniverse.symbol == symbol.upper()
        ).first()
        if record is None:
            return False

        record.last_fetch_success_at = datetime.utcnow()
        record.consecutive_fetch_failures = 0
        return True

    def record_fetch_failure(
        self,
        db: Session,
        symbol: str,
        *,
        reason: str,
        trigger_source: str,
        no_data: bool,
        deactivate_threshold: int = 3,
    ) -> Dict[str, Any]:
        """Increment failure counters and deactivate symbols that repeatedly return no data."""
        record = db.query(StockUniverse).filter(
            StockUniverse.symbol == symbol.upper()
        ).first()
        if record is None:
            return {"count": 0, "deactivated": False}

        now = datetime.utcnow()
        record.last_fetch_failure_at = now
        record.consecutive_fetch_failures = (record.consecutive_fetch_failures or 0) + 1
        deactivated = False

        if (
            no_data
            and self._normalize_status(record) == UNIVERSE_STATUS_ACTIVE
            and record.consecutive_fetch_failures >= deactivate_threshold
        ):
            deactivated = self._apply_status_transition(
                db,
                record,
                new_status=UNIVERSE_STATUS_INACTIVE_NO_DATA,
                trigger_source=trigger_source,
                reason=reason,
                now=now,
                payload={"consecutive_failures": record.consecutive_fetch_failures},
            )

        return {
            "count": record.consecutive_fetch_failures,
            "deactivated": deactivated,
        }

    def fetch_sp500_symbols(self) -> List[str]:
        """
        Fetch S&P 500 stock symbols from Wikipedia.

        Returns:
            List of S&P 500 stock symbols
        """
        try:
            logger.info("Fetching S&P 500 symbols from Wikipedia")

            # Fetch S&P 500 list from Wikipedia with User-Agent header
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

            # Use requests to fetch with User-Agent header (Wikipedia blocks requests without it)
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse HTML tables from the response content
            tables = pd.read_html(response.text)

            # First table contains the S&P 500 companies
            sp500_table = tables[0]

            # Extract symbols (in 'Symbol' column)
            symbols = sp500_table['Symbol'].tolist()

            # Clean symbols (remove any dots - some symbols may have class indicators)
            symbols = [str(s).replace('.', '-') for s in symbols]

            logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}", exc_info=True)
            return []

    def update_sp500_membership(self, db: Session) -> Dict:
        """
        Update S&P 500 membership for all stocks in the universe.

        Fetches current S&P 500 list and updates is_sp500 flag.

        Args:
            db: Database session

        Returns:
            Dict with stats: {sp500_count: int, updated: int}
        """
        try:
            # Fetch S&P 500 symbols
            sp500_symbols = self.fetch_sp500_symbols()

            if not sp500_symbols:
                logger.warning("No S&P 500 symbols fetched")
                return {'sp500_count': 0, 'updated': 0}

            # First, set all stocks to is_sp500=False
            db.query(StockUniverse).update({'is_sp500': False}, synchronize_session=False)

            # Then, set S&P 500 stocks to is_sp500=True
            updated = db.query(StockUniverse).filter(
                StockUniverse.symbol.in_(sp500_symbols)
            ).update(
                {'is_sp500': True, 'updated_at': datetime.utcnow()},
                synchronize_session=False
            )

            db.commit()

            logger.info(f"Updated S&P 500 membership: {updated} stocks marked as S&P 500")

            return {
                'sp500_count': len(sp500_symbols),
                'updated': updated
            }

        except Exception as e:
            logger.error(f"Error updating S&P 500 membership: {e}", exc_info=True)
            db.rollback()
            return {'sp500_count': 0, 'updated': 0}

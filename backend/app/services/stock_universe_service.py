"""
Stock Universe Service for managing scannable stock lists.

Fetches stocks from finviz and manages the stock_universe database table.
"""
import logging
import csv
import io
import json
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional
from finvizfinance.screener.overview import Overview
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from ..models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_STATUS_ACTIVE,
    UNIVERSE_STATUS_INACTIVE_MANUAL,
    UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
    UNIVERSE_STATUS_INACTIVE_NO_DATA,
)

logger = logging.getLogger(__name__)


class StockUniverseService:
    """Service for managing stock universe (NYSE/NASDAQ stocks from finviz)"""

    def __init__(self):
        """Initialize stock universe service."""
        pass

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
            StockUniverseStatusEvent(
                symbol=symbol,
                old_status=old_status,
                new_status=new_status,
                trigger_source=trigger_source,
                reason=reason,
                payload_json=json.dumps(payload, sort_keys=True) if payload else None,
            )
        )

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
        if changed or payload:
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
        Fetch all US stocks from finviz using finvizfinance library.

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

                        # Create new screener instance for each exchange
                        fviz = Overview()
                        fviz.set_filter(filters_dict={'Exchange': exchange})
                        df = fviz.screener_view(verbose=0)

                        if df is not None and not df.empty:
                            stocks = self._parse_finviz_dataframe(df, exchange)
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

                fviz = Overview()
                fviz.set_filter(filters_dict={'Exchange': exchange_name})
                df = fviz.screener_view(verbose=0)

                if df is None or df.empty:
                    logger.warning(f"No data returned from finviz for exchange {exchange_filter}")
                    return []

                stocks = self._parse_finviz_dataframe(df, exchange_name)
                logger.info(f"Successfully fetched {len(stocks)} stocks from {exchange_name}")
                return stocks

        except Exception as e:
            logger.error(f"Error fetching stocks from finviz: {e}", exc_info=True)
            return []

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
            stock_map = {
                stock.symbol: stock
                for stock in db.query(StockUniverse).filter(
                    StockUniverse.symbol.in_([row["symbol"] for row in stocks])
                ).all()
            }

            for stock_data in stocks:
                symbol = stock_data["symbol"]
                existing = stock_map.get(symbol)

                if existing:
                    existing.name = stock_data["name"] or existing.name
                    existing.exchange = stock_data["exchange"] or existing.exchange
                    existing.sector = stock_data["sector"] or existing.sector
                    existing.industry = stock_data["industry"] or existing.industry
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
                    new_stock = StockUniverse(
                        symbol=symbol,
                        name=stock_data["name"],
                        exchange=stock_data["exchange"],
                        sector=stock_data["sector"],
                        industry=stock_data["industry"],
                        market_cap=stock_data["market_cap"],
                        is_active=True,
                        status=UNIVERSE_STATUS_ACTIVE,
                        status_reason="Imported from CSV",
                        source="csv",
                        added_at=now,
                        first_seen_at=now,
                        updated_at=now,
                    )
                    db.add(new_stock)
                    self._add_status_event(
                        db,
                        symbol=symbol,
                        old_status=None,
                        new_status=UNIVERSE_STATUS_ACTIVE,
                        trigger_source="csv_import",
                        reason="Imported from CSV",
                        payload={"source": "csv"},
                    )
                    added_count += 1

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

        Performs upsert: updates existing symbols, inserts new ones, deactivates removed ones.

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

            now = datetime.utcnow()
            added_count = 0
            updated_count = 0

            existing_stocks = {
                stock.symbol: stock
                for stock in db.query(StockUniverse).all()
            }
            fetched_symbols = {stock_data["symbol"] for stock_data in stocks}

            for stock_data in stocks:
                symbol = stock_data["symbol"]
                existing = existing_stocks.get(symbol)
                if existing is None:
                    new_stock = StockUniverse(
                        symbol=symbol,
                        name=stock_data["name"],
                        exchange=stock_data["exchange"],
                        sector=stock_data["sector"],
                        industry=stock_data["industry"],
                        market_cap=stock_data["market_cap"],
                        is_active=True,
                        status=UNIVERSE_STATUS_ACTIVE,
                        status_reason="Present in Finviz universe sync",
                        source="finviz",
                        added_at=now,
                        first_seen_at=now,
                        last_seen_in_source_at=now,
                        updated_at=now,
                    )
                    db.add(new_stock)
                    self._add_status_event(
                        db,
                        symbol=symbol,
                        old_status=None,
                        new_status=UNIVERSE_STATUS_ACTIVE,
                        trigger_source="finviz_sync",
                        reason="New symbol discovered in Finviz universe sync",
                        payload={"exchange": stock_data["exchange"]},
                    )
                    added_count += 1
                    continue

                existing.name = stock_data["name"] or existing.name
                existing.exchange = stock_data["exchange"] or existing.exchange
                existing.sector = stock_data["sector"] or existing.sector
                existing.industry = stock_data["industry"] or existing.industry
                existing.market_cap = stock_data["market_cap"]
                self._apply_status_transition(
                    db,
                    existing,
                    new_status=UNIVERSE_STATUS_ACTIVE,
                    trigger_source="finviz_sync",
                    reason="Present in Finviz universe sync",
                    now=now,
                    payload={"exchange": stock_data["exchange"]},
                    source="finviz",
                    clear_failures=True,
                    seen_in_source=True,
                )
                updated_count += 1

            # Deactivate symbols that no longer exist in finviz
            # ONLY when refreshing ALL exchanges (no filter)
            # When refreshing specific exchange, don't deactivate stocks from other exchanges
            deactivated_count = 0

            # Safety check: minimum expected stocks from finviz (prevents mass deactivation on API failure)
            MIN_EXPECTED_STOCKS = 8000  # Finviz typically returns 9000+ US stocks
            MAX_DEACTIVATION_PERCENT = 10  # Never deactivate more than 10% in one refresh

            if not exchange_filter:
                # Only deactivate when doing full universe refresh
                removed_records = [
                    stock
                    for stock in existing_stocks.values()
                    if self._normalize_status(stock) == UNIVERSE_STATUS_ACTIVE
                    and stock.symbol not in fetched_symbols
                ]

                if removed_records:
                    # Safety check: don't deactivate if fetch count is suspiciously low
                    if len(fetched_symbols) < MIN_EXPECTED_STOCKS:
                        logger.warning(
                            f"SAFETY: Skipping deactivation - only {len(fetched_symbols)} stocks fetched "
                            f"(expected {MIN_EXPECTED_STOCKS}+). Would have deactivated {len(removed_records)} stocks."
                        )
                    # Safety check: don't deactivate too many stocks at once
                    elif len(removed_records) > len(existing_stocks) * MAX_DEACTIVATION_PERCENT / 100:
                        logger.warning(
                            f"SAFETY: Skipping deactivation - {len(removed_records)} stocks would be deactivated "
                            f"(>{MAX_DEACTIVATION_PERCENT}% of {len(existing_stocks)} existing). "
                            f"This may indicate a finviz API issue."
                        )
                    else:
                        for record in removed_records:
                            self._apply_status_transition(
                                db,
                                record,
                                new_status=UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
                                trigger_source="finviz_sync",
                                reason="Missing from Finviz universe sync",
                                now=now,
                                payload={"exchange_filter": None},
                            )
                        deactivated_count = len(removed_records)
                        logger.info(f"Deactivated {deactivated_count} stocks no longer in finviz")
            else:
                # When refreshing specific exchange, only deactivate stocks from THAT exchange
                # that are no longer in the fetch results
                exchange_name = exchange_filter.upper()
                removed_from_exchange = [
                    stock
                    for stock in existing_stocks.values()
                    if stock.exchange == exchange_name
                    and self._normalize_status(stock) == UNIVERSE_STATUS_ACTIVE
                    and stock.symbol not in fetched_symbols
                ]

                if removed_from_exchange:
                    # Safety check: don't deactivate too many from a single exchange
                    existing_exchange_count = sum(
                        1
                        for stock in existing_stocks.values()
                        if stock.exchange == exchange_name and self._normalize_status(stock) == UNIVERSE_STATUS_ACTIVE
                    )
                    if existing_exchange_count > 0 and len(removed_from_exchange) > existing_exchange_count * MAX_DEACTIVATION_PERCENT / 100:
                        logger.warning(
                            f"SAFETY: Skipping deactivation for {exchange_name} - {len(removed_from_exchange)} stocks "
                            f"would be deactivated (>{MAX_DEACTIVATION_PERCENT}% of {existing_exchange_count}). "
                            f"This may indicate a finviz API issue."
                        )
                    else:
                        for record in removed_from_exchange:
                            self._apply_status_transition(
                                db,
                                record,
                                new_status=UNIVERSE_STATUS_INACTIVE_MISSING_SOURCE,
                                trigger_source="finviz_sync",
                                reason=f"Missing from Finviz universe sync for {exchange_name}",
                                now=now,
                                payload={"exchange_filter": exchange_name},
                            )
                        deactivated_count = len(removed_from_exchange)
                        logger.info(f"Deactivated {deactivated_count} {exchange_name} stocks no longer in finviz")

            db.commit()

            logger.info(f"Universe populated: {added_count} added, {updated_count} updated, {deactivated_count} deactivated")

            return {
                'added': added_count,
                'updated': updated_count,
                'deactivated': deactivated_count,
                'total': len(stocks),
            }

        except Exception as e:
            logger.error(f"Error populating universe: {e}", exc_info=True)
            db.rollback()
            raise

    def get_active_symbols(
        self,
        db: Session,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        sector: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        sp500_only: bool = False,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get list of active symbols for scanning.

        Args:
            db: Database session
            market: Optional market filter (US, HK, JP, TW)
            exchange: Optional exchange filter (NYSE, NASDAQ, AMEX)
            sector: Optional sector filter
            min_market_cap: Optional minimum market cap filter
            sp500_only: If True, only return S&P 500 stocks
            limit: Optional limit on number of symbols

        Returns:
            List of symbol strings
        """
        try:
            query = db.query(StockUniverse.symbol).filter(
                StockUniverse.active_filter()
            )

            if sp500_only:
                query = query.filter(StockUniverse.is_sp500 == True)

            if market:
                query = query.filter(StockUniverse.market == market.upper())

            if exchange:
                query = query.filter(StockUniverse.exchange == exchange.upper())

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
                "Retrieved %d active symbols (market=%s, exchange=%s, sector=%s, sp500_only=%s)",
                len(symbols),
                market,
                exchange,
                sector,
                sp500_only,
            )
            return symbols

        except Exception as e:
            logger.error(f"Error getting active symbols: {e}", exc_info=True)
            return []

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
                'recent_deactivations': [],
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

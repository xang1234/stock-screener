"""
Stock Universe Service for managing scannable stock lists.

Fetches stocks from finviz and manages the stock_universe database table.
"""
import logging
import csv
import io
import pandas as pd
from typing import List, Dict, Optional
from finvizfinance.screener.overview import Overview
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from datetime import datetime

from ..models.stock_universe import StockUniverse
from ..database import SessionLocal

logger = logging.getLogger(__name__)


class StockUniverseService:
    """Service for managing stock universe (NYSE/NASDAQ stocks from finviz)"""

    def __init__(self):
        """Initialize stock universe service."""
        pass

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

            # Upsert stocks
            for stock_data in stocks:
                symbol = stock_data['symbol']

                # Check if symbol exists
                existing = db.query(StockUniverse).filter(StockUniverse.symbol == symbol).first()

                if existing:
                    # Update existing record
                    existing.name = stock_data['name'] or existing.name
                    existing.exchange = stock_data['exchange'] or existing.exchange
                    existing.sector = stock_data['sector'] or existing.sector
                    existing.industry = stock_data['industry'] or existing.industry
                    existing.market_cap = stock_data['market_cap'] or existing.market_cap
                    existing.is_active = True  # Reactivate if was deactivated
                    existing.source = 'csv'
                    existing.updated_at = datetime.utcnow()
                    updated_count += 1
                else:
                    # Insert new record
                    new_stock = StockUniverse(
                        symbol=symbol,
                        name=stock_data['name'],
                        exchange=stock_data['exchange'],
                        sector=stock_data['sector'],
                        industry=stock_data['industry'],
                        market_cap=stock_data['market_cap'],
                        is_active=True,
                        source='csv',
                    )
                    db.add(new_stock)
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

            added_count = 0
            updated_count = 0

            # Get all existing symbols with their IDs to avoid N+1 query problem
            existing_stocks_query = select(StockUniverse.id, StockUniverse.symbol)
            existing_stocks = {row[1]: row[0] for row in db.execute(existing_stocks_query).all()}  # symbol -> id mapping

            fetched_symbols = set([stock_data['symbol'] for stock_data in stocks])

            # Separate stocks into new and existing for bulk operations
            stocks_to_insert = []
            stocks_to_update = []
            now = datetime.utcnow()

            for stock_data in stocks:
                symbol = stock_data['symbol']

                stock_dict = {
                    'symbol': symbol,
                    'name': stock_data['name'],
                    'exchange': stock_data['exchange'],
                    'sector': stock_data['sector'],
                    'industry': stock_data['industry'],
                    'market_cap': stock_data['market_cap'],
                    'is_active': True,
                    'updated_at': now,
                }

                if symbol in existing_stocks:
                    # Existing stock - prepare for update (include id for bulk_update_mappings)
                    stock_dict['id'] = existing_stocks[symbol]
                    stocks_to_update.append(stock_dict)
                else:
                    # New stock - prepare for insert
                    stock_dict['source'] = 'finviz'
                    stock_dict['added_at'] = now
                    stocks_to_insert.append(stock_dict)

            # Bulk insert new stocks
            if stocks_to_insert:
                db.bulk_insert_mappings(StockUniverse, stocks_to_insert)
                added_count = len(stocks_to_insert)
                logger.info(f"Bulk inserted {added_count} new stocks")

            # Bulk update existing stocks
            if stocks_to_update:
                db.bulk_update_mappings(StockUniverse, stocks_to_update)
                updated_count = len(stocks_to_update)
                logger.info(f"Bulk updated {updated_count} existing stocks")

            # Deactivate symbols that no longer exist in finviz
            # ONLY when refreshing ALL exchanges (no filter)
            # When refreshing specific exchange, don't deactivate stocks from other exchanges
            deactivated_count = 0

            # Safety check: minimum expected stocks from finviz (prevents mass deactivation on API failure)
            MIN_EXPECTED_STOCKS = 8000  # Finviz typically returns 9000+ US stocks
            MAX_DEACTIVATION_PERCENT = 10  # Never deactivate more than 10% in one refresh

            if not exchange_filter:
                # Only deactivate when doing full universe refresh
                removed_symbols = set(existing_stocks.keys()) - fetched_symbols

                if removed_symbols:
                    # Safety check: don't deactivate if fetch count is suspiciously low
                    if len(fetched_symbols) < MIN_EXPECTED_STOCKS:
                        logger.warning(
                            f"SAFETY: Skipping deactivation - only {len(fetched_symbols)} stocks fetched "
                            f"(expected {MIN_EXPECTED_STOCKS}+). Would have deactivated {len(removed_symbols)} stocks."
                        )
                    # Safety check: don't deactivate too many stocks at once
                    elif len(removed_symbols) > len(existing_stocks) * MAX_DEACTIVATION_PERCENT / 100:
                        logger.warning(
                            f"SAFETY: Skipping deactivation - {len(removed_symbols)} stocks would be deactivated "
                            f"(>{MAX_DEACTIVATION_PERCENT}% of {len(existing_stocks)} existing). "
                            f"This may indicate a finviz API issue."
                        )
                    else:
                        db.query(StockUniverse).filter(
                            StockUniverse.symbol.in_(removed_symbols)
                        ).update(
                            {'is_active': False, 'updated_at': datetime.utcnow()},
                            synchronize_session=False
                        )
                        deactivated_count = len(removed_symbols)
                        logger.info(f"Deactivated {deactivated_count} stocks no longer in finviz")
            else:
                # When refreshing specific exchange, only deactivate stocks from THAT exchange
                # that are no longer in the fetch results
                exchange_name = exchange_filter.upper()

                # Get existing symbols from this exchange only
                existing_exchange_symbols = set([
                    row[0] for row in db.query(StockUniverse.symbol).filter(
                        StockUniverse.exchange == exchange_name,
                        StockUniverse.is_active == True
                    ).all()
                ])

                removed_from_exchange = existing_exchange_symbols - fetched_symbols

                if removed_from_exchange:
                    # Safety check: don't deactivate too many from a single exchange
                    if len(removed_from_exchange) > len(existing_exchange_symbols) * MAX_DEACTIVATION_PERCENT / 100:
                        logger.warning(
                            f"SAFETY: Skipping deactivation for {exchange_name} - {len(removed_from_exchange)} stocks "
                            f"would be deactivated (>{MAX_DEACTIVATION_PERCENT}% of {len(existing_exchange_symbols)}). "
                            f"This may indicate a finviz API issue."
                        )
                    else:
                        db.query(StockUniverse).filter(
                            StockUniverse.symbol.in_(removed_from_exchange),
                            StockUniverse.exchange == exchange_name
                        ).update(
                            {'is_active': False, 'updated_at': datetime.utcnow()},
                            synchronize_session=False
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
            exchange: Optional exchange filter (NYSE, NASDAQ, AMEX)
            sector: Optional sector filter
            min_market_cap: Optional minimum market cap filter
            sp500_only: If True, only return S&P 500 stocks
            limit: Optional limit on number of symbols

        Returns:
            List of symbol strings
        """
        try:
            query = db.query(StockUniverse.symbol).filter(StockUniverse.is_active == True)

            if sp500_only:
                query = query.filter(StockUniverse.is_sp500 == True)

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

            logger.info(f"Retrieved {len(symbols)} active symbols (exchange={exchange}, sector={sector}, sp500_only={sp500_only})")
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
                if not existing.is_active:
                    existing.is_active = True
                    existing.updated_at = datetime.utcnow()
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
                    source='manual',
                )
                db.add(new_stock)
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
                stock.is_active = False
                stock.updated_at = datetime.utcnow()
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
            active = db.query(StockUniverse).filter(StockUniverse.is_active == True).count()

            # Count by exchange
            by_exchange = {}
            exchanges = db.query(
                StockUniverse.exchange,
                func.count(StockUniverse.id),
            ).filter(
                StockUniverse.is_active == True
            ).group_by(StockUniverse.exchange).all()

            for exchange, count in exchanges:
                by_exchange[exchange] = count

            # Count S&P 500 members
            sp500 = db.query(StockUniverse).filter(
                StockUniverse.is_active == True,
                StockUniverse.is_sp500 == True,
            ).count()

            return {
                'total': total,
                'active': active,
                'by_exchange': by_exchange,
                'sp500': sp500,
            }

        except Exception as e:
            logger.error(f"Error getting universe stats: {e}", exc_info=True)
            return {'total': 0, 'active': 0, 'by_exchange': {}, 'sp500': 0}

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


# Global instance
stock_universe_service = StockUniverseService()

"""
Breadth Calculator Service for calculating market breadth indicators.

Calculates StockBee-style breadth metrics across all active stocks
in the universe, including daily movers, multi-period ratios, and
monthly/quarterly performance indicators.
"""
import logging
from typing import Dict, Optional, List
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from ..models.stock_universe import StockUniverse
from ..models.market_breadth import MarketBreadth
from .price_cache_service import PriceCacheService

logger = logging.getLogger(__name__)


class BreadthCalculatorService:
    """
    Service for calculating market breadth indicators.

    Processes all active stocks in the universe and calculates:
    - Daily 4%+ movers (up and down)
    - 5-day and 10-day up/down ratios
    - Monthly 25%/50% movers (21 trading days)
    - Quarterly 25% movers (63 trading days)
    - 34-day 13% movers (IBD-style)
    """

    def __init__(self, db: Session):
        """
        Initialize breadth calculator service.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.price_cache = PriceCacheService.get_instance()

    def calculate_daily_breadth(self, calculation_date: date = None) -> Dict:
        """
        Calculate and return all market breadth indicators for a given date.

        Args:
            calculation_date: Date to calculate breadth for (defaults to today)

        Returns:
            Dict with all 13 breadth indicators and metadata:
            {
                'stocks_up_4pct': int,
                'stocks_down_4pct': int,
                'ratio_5day': float or None,
                'ratio_10day': float or None,
                'stocks_up_25pct_quarter': int,
                'stocks_down_25pct_quarter': int,
                'stocks_up_25pct_month': int,
                'stocks_down_25pct_month': int,
                'stocks_up_50pct_month': int,
                'stocks_down_50pct_month': int,
                'stocks_up_13pct_34days': int,
                'stocks_down_13pct_34days': int,
                'total_stocks_scanned': int
            }
        """
        if calculation_date is None:
            from ..utils.market_hours import get_eastern_now
            calculation_date = get_eastern_now().date()

        logger.info(f"Calculating breadth indicators for {calculation_date}")

        # Get all active stocks from universe
        active_stocks = self.db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).all()

        logger.info(f"Found {len(active_stocks)} active stocks in universe")

        # Initialize counters
        metrics = {
            'stocks_up_4pct': 0,
            'stocks_down_4pct': 0,
            'stocks_up_25pct_quarter': 0,
            'stocks_down_25pct_quarter': 0,
            'stocks_up_25pct_month': 0,
            'stocks_down_25pct_month': 0,
            'stocks_up_50pct_month': 0,
            'stocks_down_50pct_month': 0,
            'stocks_up_13pct_34days': 0,
            'stocks_down_13pct_34days': 0,
            'total_stocks_scanned': 0,
            'skipped_stocks': 0
        }

        # Process stocks in batches for memory management
        batch_size = 500
        total_stocks = len(active_stocks)

        for i in range(0, total_stocks, batch_size):
            batch = active_stocks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)")

            for stock in batch:
                try:
                    stock_metrics = self._calculate_stock_metrics(
                        symbol=stock.symbol,
                        end_date=calculation_date
                    )

                    if stock_metrics is None:
                        metrics['skipped_stocks'] += 1
                        continue

                    # Count stocks meeting thresholds
                    # Daily 4%+ movers
                    if stock_metrics.get('pct_change_1d', 0) >= 4.0:
                        metrics['stocks_up_4pct'] += 1
                    elif stock_metrics.get('pct_change_1d', 0) <= -4.0:
                        metrics['stocks_down_4pct'] += 1

                    # 21-day (monthly) 25% movers
                    if stock_metrics.get('pct_change_21d', 0) >= 25.0:
                        metrics['stocks_up_25pct_month'] += 1
                    elif stock_metrics.get('pct_change_21d', 0) <= -25.0:
                        metrics['stocks_down_25pct_month'] += 1

                    # 21-day (monthly) 50% movers
                    if stock_metrics.get('pct_change_21d', 0) >= 50.0:
                        metrics['stocks_up_50pct_month'] += 1
                    elif stock_metrics.get('pct_change_21d', 0) <= -50.0:
                        metrics['stocks_down_50pct_month'] += 1

                    # 34-day 13% movers (IBD-style)
                    if stock_metrics.get('pct_change_34d', 0) >= 13.0:
                        metrics['stocks_up_13pct_34days'] += 1
                    elif stock_metrics.get('pct_change_34d', 0) <= -13.0:
                        metrics['stocks_down_13pct_34days'] += 1

                    # 63-day (quarterly) 25% movers
                    if stock_metrics.get('pct_change_63d', 0) >= 25.0:
                        metrics['stocks_up_25pct_quarter'] += 1
                    elif stock_metrics.get('pct_change_63d', 0) <= -25.0:
                        metrics['stocks_down_25pct_quarter'] += 1

                    metrics['total_stocks_scanned'] += 1

                except Exception as e:
                    logger.warning(f"Error processing {stock.symbol}: {e}")
                    metrics['skipped_stocks'] += 1
                    continue

        logger.info(f"Processed {metrics['total_stocks_scanned']} stocks, skipped {metrics['skipped_stocks']}")

        # Calculate multi-day ratios from historical breadth data
        ratios = self._calculate_ratios(calculation_date)
        metrics['ratio_5day'] = ratios.get('ratio_5day')
        metrics['ratio_10day'] = ratios.get('ratio_10day')

        return metrics

    def _calculate_stock_metrics(self, symbol: str, end_date: date) -> Optional[Dict]:
        """
        Calculate percentage changes for a single stock across multiple periods.

        Args:
            symbol: Stock ticker symbol
            end_date: End date for calculations

        Returns:
            Dict with percentage changes or None if insufficient data:
            {
                'pct_change_1d': float,
                'pct_change_21d': float,
                'pct_change_34d': float,
                'pct_change_63d': float
            }
        """
        try:
            # Get 2-year historical data (enough for all calculations)
            prices_df = self.price_cache.get_historical_data(
                symbol=symbol,
                period="2y"
            )

            if prices_df is None or prices_df.empty:
                logger.debug(f"No price data for {symbol}")
                return None

            # Filter data up to end_date
            prices_df = prices_df[prices_df.index <= pd.Timestamp(end_date)]

            if len(prices_df) < 70:  # Need at least 70 days for quarterly calculations
                logger.debug(f"Insufficient data for {symbol}: {len(prices_df)} days")
                return None

            # Calculate percentage changes for different periods
            metrics = {}

            # 1-day change
            metrics['pct_change_1d'] = self._get_price_change(prices_df, 1)

            # 21-day change (monthly)
            metrics['pct_change_21d'] = self._get_price_change(prices_df, 21)

            # 34-day change (IBD-style)
            metrics['pct_change_34d'] = self._get_price_change(prices_df, 34)

            # 63-day change (quarterly)
            metrics['pct_change_63d'] = self._get_price_change(prices_df, 63)

            return metrics

        except Exception as e:
            logger.debug(f"Error calculating metrics for {symbol}: {e}")
            return None

    def _get_price_change(self, prices_df: pd.DataFrame, days: int) -> float:
        """
        Calculate N-day percentage change from a price DataFrame.

        Args:
            prices_df: DataFrame with price data (indexed by date)
            days: Number of days to look back

        Returns:
            Percentage change over the period

        Formula:
            ((latest_close - close_N_days_ago) / close_N_days_ago) * 100
        """
        if len(prices_df) < days + 1:
            return 0.0

        try:
            # Get most recent close and close from N days ago
            latest_close = prices_df['Close'].iloc[-1]
            past_close = prices_df['Close'].iloc[-(days + 1)]

            if past_close == 0 or pd.isna(past_close) or pd.isna(latest_close):
                return 0.0

            pct_change = ((latest_close - past_close) / past_close) * 100
            return round(pct_change, 2)

        except (IndexError, KeyError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating {days}-day change: {e}")
            return 0.0

    def _calculate_ratios(self, calculation_date: date) -> Dict:
        """
        Calculate 5-day and 10-day up/down ratios from historical breadth.

        Ratios are calculated as:
        - 5-day ratio: Sum(stocks_up_4pct over 5 days) / Sum(stocks_down_4pct over 5 days)
        - 10-day ratio: Sum(stocks_up_4pct over 10 days) / Sum(stocks_down_4pct over 10 days)

        Args:
            calculation_date: Date to calculate ratios for

        Returns:
            Dict with ratios (None if insufficient data or division by zero):
            {
                'ratio_5day': float or None,
                'ratio_10day': float or None
            }
        """
        try:
            # Get last 10 trading days of breadth data (including today if it exists)
            start_date = calculation_date - timedelta(days=20)  # Generous window for trading days

            breadth_records = self.db.query(MarketBreadth).filter(
                MarketBreadth.date >= start_date,
                MarketBreadth.date < calculation_date  # Don't include today
            ).order_by(MarketBreadth.date.desc()).limit(10).all()

            if len(breadth_records) < 5:
                logger.info(f"Insufficient historical breadth data for ratios: {len(breadth_records)} days")
                return {'ratio_5day': None, 'ratio_10day': None}

            # Calculate 5-day ratio
            last_5_days = breadth_records[:5]
            up_5day = sum(r.stocks_up_4pct for r in last_5_days)
            down_5day = sum(r.stocks_down_4pct for r in last_5_days)

            ratio_5day = None
            if down_5day > 0:
                ratio_5day = round(up_5day / down_5day, 2)

            # Calculate 10-day ratio
            ratio_10day = None
            if len(breadth_records) >= 10:
                last_10_days = breadth_records[:10]
                up_10day = sum(r.stocks_up_4pct for r in last_10_days)
                down_10day = sum(r.stocks_down_4pct for r in last_10_days)

                if down_10day > 0:
                    ratio_10day = round(up_10day / down_10day, 2)

            logger.info(f"Ratios calculated: 5-day={ratio_5day}, 10-day={ratio_10day}")

            return {
                'ratio_5day': ratio_5day,
                'ratio_10day': ratio_10day
            }

        except Exception as e:
            logger.error(f"Error calculating ratios: {e}", exc_info=True)
            return {'ratio_5day': None, 'ratio_10day': None}

    def find_missing_dates(self, lookback_days: int = 30) -> List[date]:
        """
        Find missing trading dates in the market_breadth table.

        Checks the lookback window for weekdays (excluding holidays)
        that don't have breadth records.

        Args:
            lookback_days: Number of days to look back for gaps

        Returns:
            List of missing trading dates (oldest first)
        """
        from sqlalchemy import func
        from ..utils.market_hours import is_trading_day, get_eastern_now

        today = get_eastern_now().date()
        start_date = today - timedelta(days=lookback_days)

        # Get all dates that have breadth data
        existing_dates = self.db.query(
            func.distinct(MarketBreadth.date)
        ).filter(
            MarketBreadth.date >= start_date
        ).all()

        existing_date_set = {d[0] for d in existing_dates}

        # Generate all trading days in range using market calendar
        missing_dates = []
        current_date = start_date

        while current_date < today:  # Exclude today (will be calculated separately)
            if is_trading_day(current_date):
                if current_date not in existing_date_set:
                    missing_dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Found {len(missing_dates)} missing breadth dates in last {lookback_days} days")
        return sorted(missing_dates)

    def fill_gaps(self, missing_dates: List[date]) -> Dict:
        """
        Fill gaps by calculating breadth for missing dates.

        Processes dates oldest first to ensure ratio calculations
        have prior data available.

        Args:
            missing_dates: List of dates to calculate breadth for

        Returns:
            Statistics about the gap-fill operation:
            {
                'total_dates': int,
                'processed': int,
                'errors': int,
                'error_dates': List[str]
            }
        """
        if not missing_dates:
            return {
                'total_dates': 0,
                'processed': 0,
                'errors': 0,
                'error_dates': []
            }

        logger.info(f"Filling {len(missing_dates)} missing breadth dates")

        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'errors': 0,
            'error_dates': []
        }

        # Process oldest first for ratio calculation accuracy
        for calc_date in sorted(missing_dates):
            try:
                # Calculate breadth for this date
                metrics = self.calculate_daily_breadth(calculation_date=calc_date)

                if metrics and metrics.get('total_stocks_scanned', 0) > 0:
                    # Store the record
                    self._store_breadth_record(calc_date, metrics)
                    stats['processed'] += 1
                    logger.debug(f"Filled gap for {calc_date}: {metrics['total_stocks_scanned']} stocks")
                else:
                    stats['errors'] += 1
                    stats['error_dates'].append(calc_date.strftime('%Y-%m-%d'))
                    logger.warning(f"No data for {calc_date}")

            except Exception as e:
                stats['errors'] += 1
                stats['error_dates'].append(calc_date.strftime('%Y-%m-%d'))
                logger.error(f"Error filling gap for {calc_date}: {e}")

        logger.info(
            f"Gap-fill complete: {stats['processed']} processed, "
            f"{stats['errors']} errors"
        )

        return stats

    def _store_breadth_record(self, calc_date: date, metrics: Dict) -> None:
        """
        Store or update a breadth record in the database.

        Args:
            calc_date: Date of the breadth record
            metrics: Calculated breadth metrics
        """
        # Check if record already exists
        existing_record = self.db.query(MarketBreadth).filter(
            MarketBreadth.date == calc_date
        ).first()

        if existing_record:
            # Update existing record
            existing_record.stocks_up_4pct = metrics['stocks_up_4pct']
            existing_record.stocks_down_4pct = metrics['stocks_down_4pct']
            existing_record.ratio_5day = metrics.get('ratio_5day')
            existing_record.ratio_10day = metrics.get('ratio_10day')
            existing_record.stocks_up_25pct_quarter = metrics['stocks_up_25pct_quarter']
            existing_record.stocks_down_25pct_quarter = metrics['stocks_down_25pct_quarter']
            existing_record.stocks_up_25pct_month = metrics['stocks_up_25pct_month']
            existing_record.stocks_down_25pct_month = metrics['stocks_down_25pct_month']
            existing_record.stocks_up_50pct_month = metrics['stocks_up_50pct_month']
            existing_record.stocks_down_50pct_month = metrics['stocks_down_50pct_month']
            existing_record.stocks_up_13pct_34days = metrics['stocks_up_13pct_34days']
            existing_record.stocks_down_13pct_34days = metrics['stocks_down_13pct_34days']
            existing_record.total_stocks_scanned = metrics['total_stocks_scanned']
            existing_record.calculation_duration_seconds = metrics.get('calculation_duration_seconds')
            logger.debug(f"Updated existing breadth record for {calc_date}")
        else:
            # Create new record
            breadth_record = MarketBreadth(
                date=calc_date,
                stocks_up_4pct=metrics['stocks_up_4pct'],
                stocks_down_4pct=metrics['stocks_down_4pct'],
                ratio_5day=metrics.get('ratio_5day'),
                ratio_10day=metrics.get('ratio_10day'),
                stocks_up_25pct_quarter=metrics['stocks_up_25pct_quarter'],
                stocks_down_25pct_quarter=metrics['stocks_down_25pct_quarter'],
                stocks_up_25pct_month=metrics['stocks_up_25pct_month'],
                stocks_down_25pct_month=metrics['stocks_down_25pct_month'],
                stocks_up_50pct_month=metrics['stocks_up_50pct_month'],
                stocks_down_50pct_month=metrics['stocks_down_50pct_month'],
                stocks_up_13pct_34days=metrics['stocks_up_13pct_34days'],
                stocks_down_13pct_34days=metrics['stocks_down_13pct_34days'],
                total_stocks_scanned=metrics['total_stocks_scanned'],
                calculation_duration_seconds=metrics.get('calculation_duration_seconds')
            )
            self.db.add(breadth_record)
            logger.debug(f"Created new breadth record for {calc_date}")

        self.db.commit()

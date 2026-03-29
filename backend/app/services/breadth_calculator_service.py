"""
Breadth Calculator Service for calculating market breadth indicators.

Calculates StockBee-style breadth metrics across all active stocks
in the universe, including daily movers, multi-period ratios, and
monthly/quarterly performance indicators.
"""
import logging
from collections import deque
from typing import Dict, Optional, List
from datetime import date, datetime, timedelta
import pandas as pd
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

    @staticmethod
    def _empty_metrics() -> Dict:
        """Return a zeroed breadth metrics container."""
        return {
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
            'ratio_5day': None,
            'ratio_10day': None,
            'total_stocks_scanned': 0,
            'skipped_stocks': 0,
            'cache_miss_stocks': 0,
            'insufficient_data_stocks': 0,
            'error_stocks': 0,
        }

    def calculate_daily_breadth(self, calculation_date: date = None, cache_only: bool = False) -> Dict:
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
        metrics = self._empty_metrics()

        # Process stocks in batches for memory management
        batch_size = 500
        total_stocks = len(active_stocks)

        for i in range(0, total_stocks, batch_size):
            batch = active_stocks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)")
            batch_symbols = [stock.symbol for stock in batch]
            price_data_by_symbol, cache_miss_symbols = self._load_price_data_for_batch(
                batch_symbols=batch_symbols,
                cache_only=cache_only,
            )
            metrics['cache_miss_stocks'] += len(cache_miss_symbols)

            for stock in batch:
                try:
                    price_history = price_data_by_symbol.get(stock.symbol)
                    if price_history is None or price_history.empty:
                        metrics['skipped_stocks'] += 1
                        continue

                    stock_metrics = self._calculate_stock_metrics_from_prices(
                        prices_df=price_history,
                        end_date=calculation_date,
                    )

                    if stock_metrics is None:
                        metrics['insufficient_data_stocks'] += 1
                        metrics['skipped_stocks'] += 1
                        continue

                    self._apply_stock_metrics(metrics, stock_metrics)
                    metrics['total_stocks_scanned'] += 1

                except Exception as e:
                    logger.warning(f"Error processing {stock.symbol}: {e}")
                    metrics['error_stocks'] += 1
                    metrics['skipped_stocks'] += 1
                    continue

        logger.info(
            "Processed %s stocks, skipped %s (cache misses=%s, insufficient=%s, errors=%s)",
            metrics['total_stocks_scanned'],
            metrics['skipped_stocks'],
            metrics['cache_miss_stocks'],
            metrics['insufficient_data_stocks'],
            metrics['error_stocks'],
        )

        # Calculate multi-day ratios from historical breadth data
        ratios = self._calculate_ratios(calculation_date)
        metrics['ratio_5day'] = ratios.get('ratio_5day')
        metrics['ratio_10day'] = ratios.get('ratio_10day')

        return metrics

    def backfill_range(self, start_date: date, end_date: date, trading_dates: Optional[List[date]] = None) -> Dict:
        """
        Calculate and persist breadth for an entire historical range.

        Price history is loaded once per symbol batch, then reused for every
        requested trading date in chronological order.
        """
        if trading_dates is None:
            from ..utils.market_hours import is_trading_day

            trading_dates = [
                current_date
                for current_date in pd.date_range(start=start_date, end=end_date, freq="D").date
                if is_trading_day(current_date)
            ]

        ordered_dates = sorted(trading_dates)
        if not ordered_dates:
            return {
                'total_dates': 0,
                'processed': 0,
                'errors': 0,
                'error_dates': [],
            }

        start_time = datetime.now()
        active_stocks = self.db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).all()
        logger.info(
            "Backfilling breadth for %s trading days across %s active stocks",
            len(ordered_dates),
            len(active_stocks),
        )

        metrics_by_date = {calc_date: self._empty_metrics() for calc_date in ordered_dates}
        batch_size = 500
        total_stocks = len(active_stocks)

        for i in range(0, total_stocks, batch_size):
            batch = active_stocks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_stocks + batch_size - 1) // batch_size
            logger.info(
                "Backfill batch %s/%s (%s stocks)",
                batch_num,
                total_batches,
                len(batch),
            )

            batch_symbols = [stock.symbol for stock in batch]
            price_data_by_symbol, _ = self._load_price_data_for_batch(
                batch_symbols=batch_symbols,
                cache_only=False,
            )

            for stock in batch:
                try:
                    price_history = price_data_by_symbol.get(stock.symbol)
                    if price_history is None or price_history.empty:
                        continue

                    for calc_date in ordered_dates:
                        stock_metrics = self._calculate_stock_metrics_from_prices(
                            prices_df=price_history,
                            end_date=calc_date,
                        )
                        if stock_metrics is None:
                            continue

                        daily_metrics = metrics_by_date[calc_date]
                        self._apply_stock_metrics(daily_metrics, stock_metrics)
                        daily_metrics['total_stocks_scanned'] += 1
                except Exception as e:
                    logger.warning("Error processing %s in breadth backfill: %s", stock.symbol, e)
                    for calc_date in ordered_dates:
                        metrics_by_date[calc_date]['error_stocks'] += 1

        prior_counts = self._get_prior_breadth_counts(ordered_dates[0], limit=10)
        rolling_counts = deque(prior_counts, maxlen=10)

        processed_dates: list[date] = []
        error_dates: list[str] = []

        for calc_date in ordered_dates:
            metrics = metrics_by_date[calc_date]
            ratios = self._calculate_ratios_from_counts(list(rolling_counts))
            metrics['ratio_5day'] = ratios['ratio_5day']
            metrics['ratio_10day'] = ratios['ratio_10day']

            if metrics['total_stocks_scanned'] > 0:
                processed_dates.append(calc_date)
                rolling_counts.append({
                    'stocks_up_4pct': metrics['stocks_up_4pct'],
                    'stocks_down_4pct': metrics['stocks_down_4pct'],
                })
            else:
                error_dates.append(calc_date.strftime('%Y-%m-%d'))

        if processed_dates:
            total_duration_seconds = (datetime.now() - start_time).total_seconds()
            duration_per_day = round(total_duration_seconds / len(processed_dates), 2)
            for calc_date in processed_dates:
                metrics_by_date[calc_date]['calculation_duration_seconds'] = duration_per_day

            self._store_breadth_records(
                {
                    calc_date: metrics_by_date[calc_date]
                    for calc_date in processed_dates
                }
            )

        return {
            'total_dates': len(ordered_dates),
            'processed': len(processed_dates),
            'errors': len(error_dates),
            'error_dates': error_dates,
        }

    def _calculate_stock_metrics_from_prices(
        self,
        prices_df: Optional[pd.DataFrame],
        end_date: date,
    ) -> Optional[Dict]:
        """Calculate breadth metrics from an already-cached price DataFrame."""
        if prices_df is None or prices_df.empty:
            return None

        end_ts = pd.Timestamp(end_date)
        if isinstance(prices_df.index, pd.DatetimeIndex) and prices_df.index.tz is not None and end_ts.tz is None:
            end_ts = end_ts.tz_localize(prices_df.index.tz)

        latest_idx = prices_df.index.searchsorted(end_ts, side='right') - 1
        if latest_idx < 69:  # Need at least 70 days for quarterly calculations
            logger.debug(f"Insufficient cached data through {end_date}: {latest_idx + 1} days")
            return None

        metrics = {
            'pct_change_1d': self._get_price_change(prices_df, 1, latest_idx),
            'pct_change_21d': self._get_price_change(prices_df, 21, latest_idx),
            'pct_change_34d': self._get_price_change(prices_df, 34, latest_idx),
            'pct_change_63d': self._get_price_change(prices_df, 63, latest_idx),
        }
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
            prices_df = self.price_cache.get_historical_data(
                symbol=symbol,
                period="2y"
            )

            if prices_df is None or prices_df.empty:
                logger.debug(f"No price data for {symbol}")
                return None

            return self._calculate_stock_metrics_from_prices(
                prices_df=prices_df,
                end_date=end_date,
            )

        except Exception as e:
            logger.debug(f"Error calculating metrics for {symbol}: {e}")
            return None

    def _get_price_change(
        self,
        prices_df: pd.DataFrame,
        days: int,
        latest_idx: Optional[int] = None,
    ) -> float:
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
        if latest_idx is None:
            latest_idx = len(prices_df) - 1

        if latest_idx < days:
            return 0.0

        try:
            # Get most recent close and close from N days ago
            latest_close = prices_df['Close'].iloc[latest_idx]
            past_close = prices_df['Close'].iloc[latest_idx - days]

            if past_close == 0 or pd.isna(past_close) or pd.isna(latest_close):
                return 0.0

            pct_change = ((latest_close - past_close) / past_close) * 100
            return round(pct_change, 2)

        except (IndexError, KeyError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating {days}-day change: {e}")
            return 0.0

    def _apply_stock_metrics(self, metrics: Dict, stock_metrics: Dict) -> None:
        """Increment aggregate breadth counters for a single stock."""
        if stock_metrics.get('pct_change_1d', 0) >= 4.0:
            metrics['stocks_up_4pct'] += 1
        elif stock_metrics.get('pct_change_1d', 0) <= -4.0:
            metrics['stocks_down_4pct'] += 1

        if stock_metrics.get('pct_change_21d', 0) >= 25.0:
            metrics['stocks_up_25pct_month'] += 1
        elif stock_metrics.get('pct_change_21d', 0) <= -25.0:
            metrics['stocks_down_25pct_month'] += 1

        if stock_metrics.get('pct_change_21d', 0) >= 50.0:
            metrics['stocks_up_50pct_month'] += 1
        elif stock_metrics.get('pct_change_21d', 0) <= -50.0:
            metrics['stocks_down_50pct_month'] += 1

        if stock_metrics.get('pct_change_34d', 0) >= 13.0:
            metrics['stocks_up_13pct_34days'] += 1
        elif stock_metrics.get('pct_change_34d', 0) <= -13.0:
            metrics['stocks_down_13pct_34days'] += 1

        if stock_metrics.get('pct_change_63d', 0) >= 25.0:
            metrics['stocks_up_25pct_quarter'] += 1
        elif stock_metrics.get('pct_change_63d', 0) <= -25.0:
            metrics['stocks_down_25pct_quarter'] += 1

    def _load_price_data_for_batch(
        self,
        batch_symbols: List[str],
        cache_only: bool,
    ) -> tuple[Dict[str, Optional[pd.DataFrame]], List[str]]:
        """Load batch price histories once, with optional cache misses fetched a single time."""
        price_data_by_symbol = self.price_cache.get_many_cached_only(
            batch_symbols,
            period="2y",
        )
        cache_miss_symbols: List[str] = []

        if cache_only:
            return price_data_by_symbol, cache_miss_symbols

        for symbol in batch_symbols:
            price_history = price_data_by_symbol.get(symbol)
            if price_history is None or price_history.empty:
                cache_miss_symbols.append(symbol)
                price_data_by_symbol[symbol] = self._calculate_stock_history(symbol)

        return price_data_by_symbol, cache_miss_symbols

    def _calculate_stock_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch a symbol's full historical data once for reuse."""
        return self.price_cache.get_historical_data(
            symbol=symbol,
            period="2y",
        )

    def _get_prior_breadth_counts(self, start_date: date, limit: int = 10) -> List[Dict[str, int]]:
        """Fetch up to the prior 10 breadth count rows before a backfill window."""
        prior_records = self.db.query(MarketBreadth).filter(
            MarketBreadth.date < start_date
        ).order_by(MarketBreadth.date.desc()).limit(limit).all()

        ordered_records = list(reversed(prior_records))
        return [
            {
                'stocks_up_4pct': record.stocks_up_4pct,
                'stocks_down_4pct': record.stocks_down_4pct,
            }
            for record in ordered_records
        ]

    def _calculate_ratios_from_counts(self, prior_counts: List[Dict[str, int]]) -> Dict:
        """Calculate 5-day and 10-day ratios from prior daily counts."""
        if len(prior_counts) < 5:
            return {'ratio_5day': None, 'ratio_10day': None}

        last_5_days = prior_counts[-5:]
        up_5day = sum(day['stocks_up_4pct'] for day in last_5_days)
        down_5day = sum(day['stocks_down_4pct'] for day in last_5_days)
        ratio_5day = round(up_5day / down_5day, 2) if down_5day > 0 else None

        ratio_10day = None
        if len(prior_counts) >= 10:
            last_10_days = prior_counts[-10:]
            up_10day = sum(day['stocks_up_4pct'] for day in last_10_days)
            down_10day = sum(day['stocks_down_4pct'] for day in last_10_days)
            if down_10day > 0:
                ratio_10day = round(up_10day / down_10day, 2)

        return {'ratio_5day': ratio_5day, 'ratio_10day': ratio_10day}

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

    def _store_breadth_records(self, metrics_by_date: Dict[date, Dict]) -> None:
        """Store or update a group of breadth records in one transaction."""
        if not metrics_by_date:
            return

        dates = sorted(metrics_by_date.keys())
        existing_records = self.db.query(MarketBreadth).filter(
            MarketBreadth.date >= dates[0],
            MarketBreadth.date <= dates[-1],
        ).all()
        existing_by_date = {record.date: record for record in existing_records}

        for calc_date in dates:
            metrics = metrics_by_date[calc_date]
            existing_record = existing_by_date.get(calc_date)
            if existing_record:
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
            else:
                self.db.add(MarketBreadth(
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
                    calculation_duration_seconds=metrics.get('calculation_duration_seconds'),
                ))

        self.db.commit()

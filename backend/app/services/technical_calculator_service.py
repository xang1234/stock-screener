"""
Technical Calculator Service

Calculates technical indicators from cached price data, eliminating the need
to fetch these values from external APIs.

Calculates:
- RSI (14)
- ATR (14)
- SMA distances (20, 50, 200) as % from current price
- Performance metrics (week, month, quarter, half year, year, YTD)
- Volatility (week, month)
- 52-week high/low distances
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalCalculatorService:
    """
    Calculate technical indicators from price data.

    All calculations use standard formulas matching finviz outputs.
    """

    def __init__(self):
        pass

    def calculate_all(self, price_data: pd.DataFrame) -> Dict[str, Optional[float]]:
        """
        Calculate all technical indicators from price data.

        Args:
            price_data: DataFrame with OHLCV data (Date index, Open/High/Low/Close/Volume columns)

        Returns:
            Dict with calculated technical indicators
        """
        if price_data is None or price_data.empty:
            logger.warning("Empty price data provided")
            return {}

        # Ensure we have enough data
        if len(price_data) < 20:
            logger.warning(f"Insufficient price data ({len(price_data)} rows)")
            return {}

        result = {}

        # Get Close prices
        close = price_data['Close']
        current_price = close.iloc[-1]

        # RSI (14)
        rsi = self._calculate_rsi(close, period=14)
        if rsi is not None:
            result['rsi_14'] = round(rsi, 2)

        # ATR (14)
        atr = self._calculate_atr(price_data, period=14)
        if atr is not None:
            result['atr_14'] = round(atr, 2)

        # SMA distances (as percentage from current price, matching finviz format)
        sma_distances = self._calculate_sma_distances(close, current_price)
        result.update(sma_distances)

        # Performance metrics
        performance = self._calculate_performance(close)
        result.update(performance)

        # Volatility
        volatility = self._calculate_volatility(close)
        result.update(volatility)

        # 52-week range
        range_52w = self._calculate_52w_range(price_data, current_price)
        result.update(range_52w)

        # Average volume
        if 'Volume' in price_data.columns:
            result['avg_volume'] = int(price_data['Volume'].tail(50).mean())

            # Relative volume (today vs 50-day avg)
            current_volume = price_data['Volume'].iloc[-1]
            avg_vol = price_data['Volume'].tail(50).mean()
            if avg_vol > 0:
                result['relative_volume'] = round(current_volume / avg_vol, 2)

        return result

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over period
        """
        try:
            if len(close) < period + 1:
                return None

            # Calculate price changes
            delta = close.diff()

            # Separate gains and losses
            gains = delta.where(delta > 0, 0.0)
            losses = (-delta).where(delta < 0, 0.0)

            # Calculate average gain/loss using exponential moving average (Wilder's smoothing)
            avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1])

        except Exception as e:
            logger.debug(f"Error calculating RSI: {e}")
            return None

    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate ATR (Average True Range).

        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = EMA of True Range over period
        """
        try:
            if len(price_data) < period + 1:
                return None

            high = price_data['High']
            low = price_data['Low']
            close = price_data['Close']

            # Calculate True Range components
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            # True Range is the max of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR is the EMA of True Range
            atr = true_range.ewm(span=period, adjust=False).mean()

            return float(atr.iloc[-1])

        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return None

    def _calculate_sma_distances(self, close: pd.Series, current_price: float) -> Dict[str, Optional[float]]:
        """
        Calculate SMA distances as percentage from current price.

        Finviz shows these as: "5.23%" meaning price is 5.23% above SMA.
        """
        result = {}

        for period in [20, 50, 200]:
            try:
                if len(close) >= period:
                    sma = close.rolling(window=period).mean().iloc[-1]
                    if sma > 0:
                        pct_distance = ((current_price - sma) / sma) * 100
                        result[f'sma_{period}'] = round(pct_distance, 2)
            except Exception as e:
                logger.debug(f"Error calculating SMA{period}: {e}")

        return result

    def _calculate_performance(self, close: pd.Series) -> Dict[str, Optional[float]]:
        """
        Calculate performance metrics over various time periods.

        Returns percentage change from period start to current price.
        """
        result = {}
        current_price = close.iloc[-1]

        # Define periods (trading days)
        periods = {
            'perf_week': 5,
            'perf_month': 21,
            'perf_quarter': 63,
            'perf_half_year': 126,
            'perf_year': 252,
        }

        for key, days in periods.items():
            try:
                if len(close) > days:
                    past_price = close.iloc[-(days + 1)]
                    if past_price > 0:
                        pct_change = ((current_price - past_price) / past_price) * 100
                        result[key] = round(pct_change, 2)
            except Exception as e:
                logger.debug(f"Error calculating {key}: {e}")

        # YTD performance (from Jan 1 of current year)
        try:
            current_year = datetime.now().year
            # Find first trading day of the year
            year_start = close[close.index >= pd.Timestamp(f'{current_year}-01-01')]
            if len(year_start) > 0:
                ytd_start_price = year_start.iloc[0]
                if ytd_start_price > 0:
                    pct_change = ((current_price - ytd_start_price) / ytd_start_price) * 100
                    result['perf_ytd'] = round(pct_change, 2)
        except Exception as e:
            logger.debug(f"Error calculating YTD performance: {e}")

        return result

    def _calculate_volatility(self, close: pd.Series) -> Dict[str, Optional[float]]:
        """
        Calculate volatility (standard deviation of returns) for week and month.

        Finviz shows volatility as a percentage.
        """
        result = {}

        try:
            # Daily returns
            returns = close.pct_change().dropna()

            if len(returns) >= 5:
                # Weekly volatility (last 5 trading days)
                weekly_std = returns.tail(5).std()
                result['volatility_week'] = round(weekly_std * 100, 2)

            if len(returns) >= 21:
                # Monthly volatility (last 21 trading days)
                monthly_std = returns.tail(21).std()
                result['volatility_month'] = round(monthly_std * 100, 2)

        except Exception as e:
            logger.debug(f"Error calculating volatility: {e}")

        return result

    def _calculate_52w_range(self, price_data: pd.DataFrame, current_price: float) -> Dict[str, Optional[float]]:
        """
        Calculate 52-week high/low and distances.

        Returns:
            week_52_high: The 52-week high price
            week_52_low: The 52-week low price
            week_52_high_distance: % below 52-week high (negative value)
            week_52_low_distance: % above 52-week low (positive value)
        """
        result = {}

        try:
            # Get last 252 trading days (approximately 1 year)
            year_data = price_data.tail(252)

            if len(year_data) < 50:
                return result

            high_52w = year_data['High'].max()
            low_52w = year_data['Low'].min()

            result['week_52_high'] = round(high_52w, 2)
            result['week_52_low'] = round(low_52w, 2)

            if high_52w > 0:
                # Distance from 52-week high (usually negative or zero)
                distance_from_high = ((current_price - high_52w) / high_52w) * 100
                result['week_52_high_distance'] = round(distance_from_high, 2)

            if low_52w > 0:
                # Distance from 52-week low (usually positive)
                distance_from_low = ((current_price - low_52w) / low_52w) * 100
                result['week_52_low_distance'] = round(distance_from_low, 2)

        except Exception as e:
            logger.debug(f"Error calculating 52-week range: {e}")

        return result

    def calculate_batch(
        self,
        price_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Calculate technical indicators for multiple symbols.

        Args:
            price_data_dict: Dict mapping symbols to their price DataFrames

        Returns:
            Dict mapping symbols to their calculated technicals
        """
        results = {}

        for symbol, price_data in price_data_dict.items():
            try:
                results[symbol] = self.calculate_all(price_data)
            except Exception as e:
                logger.warning(f"Error calculating technicals for {symbol}: {e}")
                results[symbol] = {}

        return results

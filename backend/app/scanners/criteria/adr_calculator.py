"""
Average Daily Range (ADR) calculator.

Calculates ADR as a percentage of the current price to measure stock volatility.
Higher ADR indicates more volatile/active stocks.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ADRCalculator:
    """Calculate Average Daily Range (ADR) as percentage."""

    def calculate_adr_percent(
        self,
        price_data: pd.DataFrame,
        period: int = 20,
        min_valid_rows: int | None = None,
    ) -> Optional[float]:
        """
        Calculate Average Daily Range as percentage of current price.

        ADR measures the average daily price movement as a percentage,
        providing insight into stock volatility. Higher ADR indicates
        more volatile stocks.

        Args:
            price_data: DataFrame with High, Low, Close columns (chronological order)
            period: Number of days to average (default: 20)

        Returns:
            Average daily range as percentage (e.g., 3.5 means 3.5% daily range)
            Returns None if insufficient data (< period days)

        Example:
            If stock has High=$105, Low=$100, Close=$102:
            Daily range = (105-100)/102 * 100 = 4.9%
            ADR = average of last 20 daily range percentages
        """
        try:
            # Validate required columns
            required_cols = ['High', 'Low', 'Close']
            if not all(col in price_data.columns for col in required_cols):
                logger.warning(f"Missing required columns for ADR calculation. Need {required_cols}")
                return None

            # Check sufficient data
            if len(price_data) < period:
                logger.warning(f"Insufficient data for ADR: {len(price_data)} days < {period} required")
                return None

            # Get most recent 'period' days
            recent_data = price_data.tail(period)

            valid_rows = (
                recent_data[required_cols].notna().all(axis=1)
                & (recent_data["Close"] > 0)
                & (recent_data["High"] >= recent_data["Low"])
            )
            daily_ranges_pct = (
                (recent_data.loc[valid_rows, "High"] - recent_data.loc[valid_rows, "Low"])
                / recent_data.loc[valid_rows, "Close"]
            ) * 100

            # Return None if we don't have enough valid days
            required_valid_rows = (
                min_valid_rows if min_valid_rows is not None else period * 0.8
            )
            if len(daily_ranges_pct) < required_valid_rows:  # Allow 20% missing data by default
                logger.warning(f"Too many invalid days: {len(daily_ranges_pct)} valid out of {period}")
                return None

            # Calculate and return average
            adr_percent = np.mean(daily_ranges_pct)

            return round(float(adr_percent), 2)

        except Exception as e:
            logger.error(f"Error calculating ADR: {e}", exc_info=True)
            return None

    def calculate_adr_absolute(
        self,
        price_data: pd.DataFrame,
        period: int = 20
    ) -> Optional[float]:
        """
        Calculate Average Daily Range as absolute dollar amount.

        Args:
            price_data: DataFrame with High, Low columns
            period: Number of days to average (default: 20)

        Returns:
            Average daily range in dollars or None
        """
        try:
            # Validate required columns
            if 'High' not in price_data.columns or 'Low' not in price_data.columns:
                return None

            # Check sufficient data
            if len(price_data) < period:
                return None

            # Get most recent 'period' days
            recent_data = price_data.tail(period).copy()

            # Calculate daily ranges (High - Low)
            daily_ranges = recent_data['High'] - recent_data['Low']

            # Remove invalid values
            daily_ranges = daily_ranges.dropna()
            daily_ranges = daily_ranges[daily_ranges >= 0]

            if len(daily_ranges) < period * 0.8:
                return None

            # Calculate average
            adr_absolute = daily_ranges.mean()

            return round(float(adr_absolute), 2)

        except Exception as e:
            logger.error(f"Error calculating absolute ADR: {e}", exc_info=True)
            return None
